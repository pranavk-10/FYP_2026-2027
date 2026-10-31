// Backend address. Override with VITE_API_URL in frontend/.env (see .env.example).
export const API_BASE_URL = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

/** True when the FastAPI backend answers on its root route. */
export async function checkBackend() {
  try {
    const res = await fetch(`${API_BASE_URL}/`, { method: 'GET' });
    if (!res.ok) return false;
    const data = await res.json();
    return data?.status === 'online';
  } catch {
    return false;
  }
}

/** Runs the image models, the report reader and the panel discussion in one request. */
export async function analyzeCase(files, { maxRounds = 2, gradcam = false } = {}) {
  const form = new FormData();
  if (files.ecg) form.append('ecg_file', files.ecg);
  if (files.xray) form.append('xray_file', files.xray);
  if (files.skin) form.append('skin_file', files.skin);
  if (files.blood) form.append('blood_report_file', files.blood);
  form.append('max_rounds', String(maxRounds));
  form.append('gradcam', gradcam ? 'true' : 'false');

  let res;
  try {
    res = await fetch(`${API_BASE_URL}/api/analyze/multimodal`, { method: 'POST', body: form });
  } catch {
    throw new Error(`Could not reach the backend at ${API_BASE_URL}. Check that it is running.`);
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `The backend returned an error (HTTP ${res.status}).`);
  }
  return res.json();
}
