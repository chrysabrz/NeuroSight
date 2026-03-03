# Deploy NeuroSight on Render

Follow these steps to deploy the NeuroSight dashboard (with Knowledge Base) on Render as a Web Service.

---

## 1. Push your code to GitHub

- Create a new repository on GitHub (if you don’t have one).
- From your project folder, init git (if needed), add files, and push:

```bash
cd "neurosight lit"
git init
git add neurosight_dashboard_5.py neurosight_app_final.py pages/ .streamlit/ requirements.txt
git commit -m "NeuroSight app for Render"
git branch -M main
git remote add origin https://github.com/chrysabrz/NeuroSight.git
git push -u origin main
```

- Do **not** commit `.env` or any file with secrets. Add to `.gitignore`:

```
.env
*.json
!pages/
```

- Commit and push `.gitignore` if you created or changed it.

---

## 2. Create a Web Service on Render

1. Go to [render.com](https://render.com) and sign in (or sign up with GitHub).
2. Click **New** → **Web Service**.
3. Connect your GitHub account if asked, then select the repository that contains the NeuroSight code.
4. Use these settings:

| Field | Value |
|--------|--------|
| **Name** | `neurosight` (or any name you like) |
| **Region** | Choose the closest to your users |
| **Branch** | `main` (or your default branch) |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `streamlit run neurosight_dashboard_5.py --server.port=$PORT --server.address=0.0.0.0` |
| **Instance Type** | Free (or paid if you need more resources) |

5. Click **Advanced** and add environment variables if you use the AI / OpenAI features:

| Key | Value | Secret? |
|-----|--------|--------|
| `OPENAI_API_KEY` | Your OpenAI API key | Yes ✓ |

6. Click **Create Web Service**.

Render will clone the repo, run the build command, then start the app with the start command. The first deploy may take a few minutes.

---

## 3. After deploy

- Render will show a URL like `https://neurosight-xxxx.onrender.com`.
- Open it: you should see the NeuroSight dashboard (main view).
- Use the sidebar to open **Knowledge Base** (multi-page app).
- The Knowledge Base loads data from the Google Drive URL defined in `neurosight_app_final.py`; no extra config is needed if that file is publicly accessible.

---

## 4. Optional: `render.yaml` (Blueprint)

You can define the same Web Service in `render.yaml` in the repo root and use **Blueprint** when creating the service so Render reads the build/start commands from the file.

**`render.yaml`** (create in project root):

```yaml
services:
  - type: web
    name: neurosight
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run neurosight_dashboard_5.py --server.port=$PORT --server.address=0.0.0.0
```

Then in the Render dashboard: **New** → **Blueprint**, connect the repo, and add `OPENAI_API_KEY` in the service’s Environment tab if needed.

---

## 5. Troubleshooting

- **App not loading**  
  Check the **Logs** tab on Render. Ensure the start command is exactly:
  `streamlit run neurosight_dashboard_5.py --server.port=$PORT --server.address=0.0.0.0`

- **KB or data missing**  
  The app expects the Knowledge Base JSON from a Google Drive URL. If that URL is not public or changes, update `KB_DRIVE_URL` in `neurosight_app_final.py` (or point to another public URL).

- **OpenAI / AI features not working**  
  Set `OPENAI_API_KEY` in the Render service **Environment** and redeploy. The app already uses `os.getenv("OPENAI_API_KEY")` and will work once the variable is set.

- **Free instance sleeps**  
  On the free tier, the service sleeps after inactivity. The first request after that may take 30–60 seconds to wake the instance.
