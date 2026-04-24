import io, json, base64, torch, uvicorn, numpy as np
from typing import List, Optional
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torchvision import transforms
from torchvision.utils import make_grid
from model import Generator, Critic

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

@app.get("/")
def root():
    return {"status": "API running"}

@app.get("/health")
def health():
    return {"status": "ok"}

# load saved weights
gen = Generator(z_dim=100, features=64).to(device)
critic = Critic(features=64).to(device)
gen.load_state_dict(torch.load("generator.pth", map_location=device))
critic.load_state_dict(torch.load("critic.pth", map_location=device))
gen.eval()
critic.eval()

# ── helpers ──────────────────────────────────────────────────────────────────

def tensor_to_base64(tensor):
    """Convert a single image tensor to base64 PNG."""
    img = (tensor.squeeze(0) * 0.5 + 0.5).clamp(0, 1)
    buf = io.BytesIO()
    transforms.ToPILImage()(img.cpu()).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def grid_to_base64(tensor, nrow=8):
    """Convert a batch of images to a single grid base64 PNG."""
    grid = make_grid(tensor * 0.5 + 0.5, nrow=nrow, padding=2).clamp(0, 1)
    buf = io.BytesIO()
    transforms.ToPILImage()(grid.cpu()).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def slerp(z1, z2, t):
    """Spherical linear interpolation between two z vectors."""
    z1_n = z1 / z1.norm(dim=-1, keepdim=True)
    z2_n = z2 / z2.norm(dim=-1, keepdim=True)
    omega = torch.acos((z1_n * z2_n).sum(dim=-1, keepdim=True).clamp(-1, 1))
    sin_omega = torch.sin(omega)
    # fallback to lerp when vectors are nearly parallel
    if sin_omega.abs().item() < 1e-6:
        return (1 - t) * z1 + t * z2
    return (torch.sin((1 - t) * omega) / sin_omega) * z1 + (torch.sin(t * omega) / sin_omega) * z2

# ── request models ───────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    num_images: int = 16
    truncation: float = 1.0
    z: Optional[List[float]] = None

class InterpolateRequest(BaseModel):
    steps: int = 8

# ── endpoints ────────────────────────────────────────────────────────────────

@app.get("/logs")
def get_logs():
    try:
        with open("loss_log.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"critic_loss": [], "gen_loss": []}

@app.get("/model-info")
def get_model_info():
    def count_params(model):
        return sum(p.numel() for p in model.parameters())

    def layer_info(model, name):
        layers = []
        for n, m in model.named_modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d,
                              torch.nn.BatchNorm2d, torch.nn.LeakyReLU,
                              torch.nn.ReLU, torch.nn.Tanh)):
                layers.append({
                    "name": n or name,
                    "type": m.__class__.__name__,
                    "params": sum(p.numel() for p in m.parameters()),
                    "details": str(m).split("(", 1)[-1].rstrip(")")
                })
        return layers

    return {
        "generator": {
            "total_params": count_params(gen),
            "layers": layer_info(gen, "Generator"),
            "input": "z ∈ ℝ¹⁰⁰ ~ N(0, 1)",
            "output": "RGB image 32×32"
        },
        "critic": {
            "total_params": count_params(critic),
            "layers": layer_info(critic, "Critic"),
            "input": "RGB image 32×32",
            "output": "Scalar (Wasserstein score)"
        },
        "z_dim": 100,
        "image_size": 32,
        "training_info": {
            "dataset": "CIFAR-10",
            "optimizer": "Adam (β₁=0.0, β₂=0.9)",
            "lr": "1e-4",
            "lambda_gp": 10,
            "n_critic": 5
        }
    }

@app.post("/generate")
def generate_images(req: GenerateRequest):
    if req.z is not None:
        n = 1
        trunc = req.truncation
        z = torch.tensor(req.z, dtype=torch.float32).view(1, 100, 1, 1).to(device)
    else:
        n = max(1, min(req.num_images, 64))
        trunc = max(0.1, min(req.truncation, 2.0))
        z = torch.randn(n, 100, 1, 1, device=device) * trunc
    with torch.no_grad():
        fake = gen(z)
        scores = critic(fake).tolist()

    nrow = min(n, 8)
    grid_b64 = grid_to_base64(fake, nrow=nrow)
    individual = [tensor_to_base64(fake[i]) for i in range(n)]

    return {
        "grid": grid_b64,
        "images": individual,
        "scores": [round(s, 4) for s in scores],
        "avg_score": round(sum(scores) / len(scores), 4),
        "num_generated": n,
        "truncation": trunc
    }

@app.post("/interpolate")
def interpolate_images(req: InterpolateRequest):
    steps = max(3, min(req.steps, 20))

    z1 = torch.randn(1, 100, device=device)
    z2 = torch.randn(1, 100, device=device)

    images = []
    with torch.no_grad():
        for i in range(steps):
            t = i / (steps - 1)
            z_interp = slerp(z1, z2, t).view(1, 100, 1, 1).to(device)
            fake = gen(z_interp)
            images.append(tensor_to_base64(fake))

    # also create a combined strip
    all_z = []
    for i in range(steps):
        t = i / (steps - 1)
        all_z.append(slerp(z1, z2, t).view(1, 100, 1, 1))
    z_batch = torch.cat(all_z, dim=0).to(device)
    with torch.no_grad():
        strip = gen(z_batch)
    strip_b64 = grid_to_base64(strip, nrow=steps)

    print(f"DEBUG SLERP: Computed interpolation over {steps} steps.")
    print(f"DEBUG SLERP: Emitting exactly {len(images)} independent frames.")

    return {
        "images": images,
        "strip": strip_b64,
        "steps": steps
    }

if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)