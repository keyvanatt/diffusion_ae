"""
transient/app.py — Visualisation Streamlit : theta -> champ CH4 transitoire U(t)
Usage : streamlit run transient/app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import time
import numpy as np
import matplotlib
import torch
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

from transient.main import load_model, run_inference

# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------
THETA_PARAMS = [
    ('k (Diffusion)', 0, 1.00,   0.55),
    ('A (angle)',0.0,  350.0,  175.0),
    ('C (Injection rate (kg/m3/s))', 0.005, 0.02,  0.0125),
]

MODELS = {
    'Baseline':  'checkpoints/LaplaceLatentModel.pt',
    'Finetuned': 'checkpoints/LaplaceLatentModel_finetuned.pt',
    'Corrected': 'checkpoints/CorrectionAE_best.pt',
}

CKPT_K_MAX = 20   # fréquences tronquées pour accélérer l'inférence
ANIM_MS    = 80   # ms par pas de temps réel

# Correspondance noms display → noms matplotlib
_CMAP_MAP = {
    'RdBu':    'RdBu_r',
    'Viridis': 'viridis',
    'Plasma':  'plasma',
    'Inferno': 'inferno',
    'Hot':     'hot',
    'Turbo':   'turbo',
}


@st.cache_resource
def get_model(model_key: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, ckpt = load_model(MODELS[model_key], device)
    return model, ckpt, device


@st.cache_data(show_spinner=False)
def run_cached_inference(model_key: str, k: float, A: float, C: float, dt: float):
    model, ckpt, device = get_model(model_key)
    theta = np.array([[k, A, C]], dtype=np.float32)
    return run_inference(theta, model, ckpt, device, dt=dt, k_max=CKPT_K_MAX)[0]  # (Nt, N, N)


@st.cache_data(show_spinner=False)
def make_gif_bytes(U: np.ndarray, cmap_name: str) -> bytes:
    U_disp = U
    duration_ms = ANIM_MS

    vmin, vmax = 0.0, 0.05
    U_norm = (U_disp - vmin) / max(vmax - vmin, 1e-8)

    cmap_fn = matplotlib.colormaps[_CMAP_MAP.get(cmap_name, 'viridis')]
    rgba = (cmap_fn(U_norm) * 255).astype(np.uint8)
    imgs = [Image.fromarray(f[..., :3]) for f in rgba]

    buf = io.BytesIO()
    imgs[0].save(buf, format='GIF', save_all=True, append_images=imgs[1:],
                 duration=duration_ms, optimize=False)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
st.set_page_config(page_title='CH4 Transitoire', layout='wide')
st.title('CH4 Transitoire — Surrogate Laplace AE')

# Sidebar
st.sidebar.header('Modèle')
_model_names = list(MODELS.keys())
model_key = st.sidebar.radio('Pipeline', _model_names,
                              index=_model_names.index('Corrected'))

st.sidebar.markdown('---')
st.sidebar.header('Paramètres physiques θ')
slider_vals = []
for label, lo, hi, default in THETA_PARAMS:
    v = st.sidebar.slider(label, float(lo), float(hi), float(default),
                          step=float((hi - lo) / 200))
    slider_vals.append(v)
k_val, A_val, C_val = slider_vals

dt_val = 1.0

st.sidebar.markdown('---')
_cmaps = list(_CMAP_MAP.keys())
cmap_name = st.sidebar.selectbox('Colormap', _cmaps, index=0)

# ---------------------------------------------------------------------------
# Debounce : ne relancer l'inférence qu'après 0.6 s sans changement
# ---------------------------------------------------------------------------
DEBOUNCE_S = 0.6
current_params = (model_key, k_val, A_val, C_val, cmap_name)

if 'last_params' not in st.session_state:
    st.session_state.last_params = current_params
    st.session_state.changed_at  = time.time()

if current_params != st.session_state.last_params:
    st.session_state.last_params = current_params
    st.session_state.changed_at  = time.time()

wait = DEBOUNCE_S - (time.time() - st.session_state.changed_at)
if wait > 0:
    time.sleep(wait)
    st.rerun()

# ---------------------------------------------------------------------------
# Inférence
# ---------------------------------------------------------------------------
with st.spinner('Inférence en cours…'):
    U_pred = run_cached_inference(model_key, k_val, A_val, C_val, dt_val)

Nt = U_pred.shape[0]
fps_display = 1000.0 / ANIM_MS
st.caption(
    f'θ = (k={k_val:.3f}, A={A_val:.1f}, C={C_val:.4f})  |  '
    f'shape {U_pred.shape}  |  '
    f'min `{U_pred.min():.4f}`  max `{U_pred.max():.4f}`  |  '
    f'{fps_display:.1f} fps'
)

# ---------------------------------------------------------------------------
# Vidéo streamée
# ---------------------------------------------------------------------------
with st.spinner('Encodage GIF…'):
    gif_bytes = make_gif_bytes(U_pred, cmap_name)
    st.image(gif_bytes, width=300)

# ---------------------------------------------------------------------------
# Courbe temporelle (expander)
# ---------------------------------------------------------------------------
with st.expander('Évolution temporelle (moyenne spatiale)', expanded=False):
    mean_t = U_pred.mean(axis=(1, 2))
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=list(range(Nt)), y=mean_t.tolist(),
        mode='lines', line=dict(color='royalblue'),
    ))
    fig_ts.update_layout(
        xaxis_title='t', yaxis_title='mean(U)',
        height=240,
        margin=dict(l=10, r=10, t=10, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_ts, width='stretch')
