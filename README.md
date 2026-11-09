# Wasserstein GAN

A PyTorch Wasserstein GAN demo with a FastAPI backend and React/Vite client.

## What it does

`backend/model.py` defines a generator and critic. `backend/main.py` exposes image-generation and latent-interpolation routes, along with model information and log endpoints.

The backend loads `generator.pth` and `critic.pth` from its working directory at startup. The repository also contains `loss_log.json` and a CIFAR-10 training notebook. These artifacts are not independent validation of model quality.

## Run the API

From `backend/` in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --reload
```

On macOS or Linux, use `.venv/bin/python` instead of `.\.venv\Scripts\python.exe`.

In a second PowerShell terminal, from `frontend/`:

```powershell
npm ci
$env:VITE_API_URL = "http://127.0.0.1:8000"
npm run dev
```

On macOS or Linux, start the client with `VITE_API_URL=http://127.0.0.1:8000 npm run dev`.

The commands match the source configuration. Installation, checkpoint compatibility, and runtime behavior were not retested in this documentation pass. The legacy frontend README and test file do not establish a passing current Vite test suite.
