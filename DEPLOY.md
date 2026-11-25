# Deploying to Vercel

Step by step, for the Hobby (free) plan. You do the deploying; nothing secret goes in any file.
Facts about Vercel come from its docs as read on Nov 25, 2026; menu names can drift, so if a
label differs, look for the nearest match.

## What gets deployed

- `app.py` defines `app = FastAPI(...)`. Vercel detects it as the entry point without configuration.
- Python **3.12**, from `.python-version`.
- Dependencies from `requirements.txt` (the app only: no torch, Ultralytics or Gradio).
- `vercel.json`:
  - function region `sin1` (Singapore, AWS `ap-southeast-1`), the same region as every Neon branch;
  - `maxDuration` 60 s;
  - `excludeFiles` keeps `best.pt`, tests, tools, notebooks, outputs and data out of the bundle.
- Measured locally: about 419 MB uncompressed (the Python limit is 500 MB).

## 0. Before you start

1. **Two Neon connection strings, both pooled.** Neon Console → your project → Connect → choose the branch → turn on **Connection pooling** → copy. The host contains `-pooler`.
   - **Production:** the `main` branch.
   - **Preview:** the `dev` branch, so preview tests never write to production.
2. **Two different `RATE_LIMIT_SALT` values,** one per environment:
   ```
   python -c "import secrets; print(secrets.token_hex(32))"
   ```
   Add a third one to your local `.env` as `RATE_LIMIT_SALT=…`. Without it, the local app refuses analyses with a clear 500.
3. **The other values** you already have: `LLM_API_KEY` and `DISCORD_WEBHOOK_URL`.

## 1. Import the repository

1. Go to https://vercel.com/new → **Import Git Repository** → connect GitHub if asked → choose `farzeen-khaan/ppe-safety-checker`.
2. On the configure screen:
   - **Framework Preset:** FastAPI if detected. Otherwise **Other**: zero-config detection of `app.py` still applies.
   - **Root Directory:** leave as the repository root (`./`). The repository root *is* the app.
   - **Build and Output Settings:** leave the defaults. There is no build command; Vercel installs `requirements.txt`.
3. **Environment Variables:** add them here (section 2) before pressing Deploy, or add them afterwards and redeploy.
4. **Deploy.**
   - This first deployment is of the production branch, `main`. **Expect it to fail.** `main` still has the old Gradio app, and its `requirements.txt` includes torch.
   - It changes nothing, and it disappears once `phase-8b` is merged. If it shows an error about size or dependencies, that's why.

## 2. Environment variables and their scopes

Project → **Settings → Environment Variables**. Mark each one **Secret** (or Sensitive) where offered.

| Variable | Production | Preview | Development |
| --- | --- | --- | --- |
| `DATABASE_URL` | Neon **main**, pooled | Neon **dev**, pooled | — |
| `LLM_API_KEY` | your key | same key | — |
| `DISCORD_WEBHOOK_URL` | alerts webhook | same webhook (only used if alerts are on) | — |
| `ALERTS_ENABLED` | `true` | `false` (set `true` only while testing alerts on a preview) | — |
| `RATE_LIMIT_SALT` | salt #1 | salt #2 | — |
| `LLM_MODEL` | optional | optional | — |

- **Development:** leave empty. It's only read by `vercel dev` / `vercel env pull`; locally the app reads `.env`.
- **Not needed on Vercel:** `DATABASE_URL_TEST`, `DATABASE_URL_MAIN` and `DISCORD_DIGEST_WEBHOOK_URL`. The digest runs on GitHub Actions with its own secrets.
- **Preview values apply to all non-production branches** unless you limit one to a branch (the docs allow per-branch Preview values). One set for every preview is fine here.
- **Changing a variable does not change existing deployments.** Redeploy (Deployments → ⋯ → Redeploy) to apply it.

## 3. Check the region and the protection settings

1. **Settings → Functions → Function Regions** should show **Singapore (sin1)**, set by `vercel.json`.
   - Hobby allows one region. A plan that disallows the region fails the deployment before the build.
2. **Settings → Deployment Protection.** With **Vercel Authentication** under **Standard Protection**:
   - **Preview URLs, and the deployment-specific production URLs, require a Vercel login.** An incognito window therefore can't test a preview.
   - **The production domain (`<project>.vercel.app`) should be public.** Standard Protection "protects all domains except production domains".
   - **If the production domain also asks you to log in,** protection is set wider than Standard. Switch it to Standard Protection. Don't turn protection off for previews.

## 4. Deploy and test the preview (phase-8b)

1. **Get a preview built:** pushing to `phase-8b` (for example the Part B commit) creates a **Preview** deployment automatically once the project is connected.
2. **Open the preview URL in your normal browser, logged in to Vercel** (protected, see 3.2), and check:
   - `/api/health` → `{"status":"ok","database":"ok"}`. If it says `unreachable`, wait 5 s and retry: Neon dev may be waking.
   - `/docs` loads.
   - `/` → analyse one photo from the dataset → verdicts, then the summary card.
   - **The row lands in the Neon dev branch, not production.**
3. **Send me the preview URL.** I can check the API from here only if it's reachable without a login: either a **shareable link** (one per Hobby account) or a temporary bypass. Otherwise I'll give you the exact checks to run.

## 5. Production

1. Merge `phase-8b` into `main` (pull request). The push to `main` creates the **Production** deployment.
2. On its first request, the app applies any pending migrations to Neon **main** (3–5). The daily digest may already have applied 3–4.
3. Send me the production URL (`https://<project>.vercel.app`). The exit check, including the **incognito test with a cold start,** runs there, because the production domain isn't behind Vercel Authentication.

## 6. Reading the logs

- **Build log:** Project → **Deployments** → select the deployment → the **Building** section.
  - Dependency installation and any bundle-size error appear here.
  - **If the build fails on size, don't change dependencies.** Send me the size it reports: the local measurement was 419 MB against a 500 MB limit, and the next step is a decision, not a guess.
- **Function (runtime) logs:** Project → **Logs**, or the deployment → **Logs / Runtime Logs**. Filter by status code or path.
  - Python tracebacks, 500s (for example "The server is not configured correctly: RATE_LIMIT_SALT…") and slow requests show here.
  - **Secrets never appear:** errors name variables, never values, and Discord URLs are redacted.
- **Usage:** Project → **Usage** shows invocations, duration and memory. That's useful for cold starts.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| 500 "not configured correctly: RATE_LIMIT_SALT…" | Salt missing or shorter than 32 characters in this environment | Add it (Production or Preview), redeploy |
| 503 "waking up or unreachable" on every request | Wrong `DATABASE_URL` for this scope, or not pooled | Check the scope and the `-pooler` host, redeploy |
| Summaries always "standard" | `LLM_API_KEY` missing in this scope | Add it, redeploy |
| No Discord alert | `ALERTS_ENABLED` isn't `true` in this scope (Preview defaults to `false`) | Expected on previews; set `true` for Production |
| The page loads but asks for a Vercel login | Deployment Protection on that URL | Use your logged-in browser, or the production domain |
