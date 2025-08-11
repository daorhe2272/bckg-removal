# Fondastic To‑Dos (Prioritized)

## Connectivity and Env Setup (must have)
- [x] Backend CORS: allow Vite origin `http://localhost:5173` via `backend/.env` (`ALLOWED_ORIGINS=["http://localhost:5173","http://127.0.0.1:5173"]`).
- [x] Frontend env: support base URL via `VITE_API_URL` and add `frontend/.env.example` with `VITE_API_URL=http://localhost:8000`.
- [x] Ensure local run workflow documented for PowerShell (Functions → Backend → Frontend) with ports 7071/8000/5173.

## Azure Functions: implement background removal (must have)
- [x] Add `functions/health` HTTP function (`GET /api/health`) returning `{ status: "ok" }`.
- [ ] Implement real background removal in `functions/background_removal`:
  - [ ] Decode base64 → image (Pillow).
  - [ ] Load ONNX model by `model_type` (`u2net`, `isnet`, `u2netp`) using onnxruntime.
  - [ ] Support model weights loading from Azure Blob (Azurite in dev) with local disk cache; reuse loaded session across invocations.
  - [ ] Generate alpha matte, composite to transparent PNG, return data URL.
  - [ ] Enforce size/time limits from env; return structured errors on timeout/invalid input.
- [ ] Model artifacts management:
  - [ ] Provide script/notebook to fetch ONNX weights and upload to Blob container `models/` with canonical names.
  - [ ] Document expected blob paths and local fallback for offline dev.

## Backend FastAPI improvements (must have)
- [ ] Centralize and validate supported models (`u2net`, `isnet`, `u2netp`) and max file size from settings.
- [ ] Rate limiting to protect `/api/process/image` (e.g., per‑IP limits) and request size limits.
- [ ] Structured logging with request IDs and error responses.
- [ ] Tests:
  - [ ] Unit: `/api/models` schema, validation utilities.
  - [ ] Unit: `/api/process/image` with mocked Functions call (success, 4xx, 5xx, timeout).
  - [ ] Integration: `/api/health` including Functions health probe.

## Frontend UX wiring (should have)
- [ ] Model selector UI: fetch `GET /api/models`, allow selecting model before processing.
- [ ] Health/Status indicator: use `GET /api/process/status` to signal Functions availability.
- [ ] Error states: show backend/Functions error messages with actionable tips.
- [ ] Add `data-cy` attributes for E2E selectors on key elements (dropzone, process button, results, download button). [[memory:2699590]]

## Docker & Dev Environment (should have)
- [ ] Fix `docker-compose.yml`:
  - [ ] Frontend service: use Vite port `5173:5173` and env `VITE_API_URL` (not `REACT_APP_API_URL`).
  - [ ] Provide `frontend/Dockerfile.dev` (Vite dev server) or remove frontend from compose and document local-run alternative.
  - [ ] Ensure backend depends_on Functions; set `FUNCTIONS_BASE_URL=http://functions:7071`.
- [ ] Backend Dockerfile: production build should copy frontend `dist/` to `backend/static/`.

## Build & Release (should have)
- [ ] Frontend production build pipeline: `npm run build` → copy `dist` → `backend/static`.
- [ ] .github workflows:
  - [ ] CI: lint + tests (frontend/backend) on PRs.
  - [ ] Build: backend image; Functions package check.
  - [ ] Optional: deploy jobs wired to branch conventions.

## Documentation (should have)
- [ ] Update README quickstart with Windows PowerShell commands, correct ports, and env variables.
- [ ] Add `.env.example` files for `backend/` and `frontend/`; ensure Functions `local.settings.json` is documented.
- [ ] Document model acquisition/upload process and local caching strategy.

## Monitoring & Security (nice to have for MVP+)
- [ ] Application Insights/structured logging hooks for Functions and Backend.
- [ ] Content-type sniffing and basic image safety checks server-side.
- [ ] Basic audit log of processing requests (non‑PII).

## E2E tests (nice to have for MVP+)
- [ ] Set up Cypress with a minimal suite: upload → process → download happy path; error path when Functions down. Add CI step.

## Acceptance criteria for MVP
- [ ] Uploading an image in the frontend calls backend and returns a PNG with transparent background produced by ONNX model via Functions within configured timeout.
- [ ] Users can choose model; errors are clearly shown; status indicates processing service health.
- [ ] Local dev works with three processes and documented env; CORS passes from Vite (5173).
- [ ] Basic unit/integration tests pass in CI.


