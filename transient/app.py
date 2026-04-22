"""
transient/app.py — Visualisation Streamlit : theta -> champ CH4 transitoire U(t)
Usage : streamlit run transient/app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import numpy as np
import matplotlib
import torch
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

from transient.main import load_model, run_inference

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
THETA_PARAMS = [
    ('k (Diffusion)',                  0.0,   1.00,   0.55,   '%.3f'),
    ('A (angle)',                      0.0,   350.0,  175.0,  '%.1f'),
    ('C (Injection rate (kg/m3/s))',   0.005, 0.02,   0.0125, '%.4f'),
]

MODELS = {
    'Baseline':  'checkpoints/LaplaceLatentModel.pt',
    'Finetuned': 'checkpoints/LaplaceLatentModel_finetuned.pt',
    'Corrected': 'checkpoints/CorrectionAE_best.pt',
}

CKPT_K_MAX = 20
NT         = 150

_CMAP_MAP = {
    'RdBu':    'RdBu_r',
    'Viridis': 'viridis',
    'Plasma':  'plasma',
    'Inferno': 'inferno',
    'Hot':     'hot',
    'Turbo':   'turbo',
}

dt_val = 1.0


# ---------------------------------------------------------------------------
# Fonctions cachées
# ---------------------------------------------------------------------------
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


def render_frame(frame: np.ndarray, cmap_name: str) -> bytes:
    vmin, vmax = 0.0, 0.05
    U_norm  = (frame - vmin) / max(vmax - vmin, 1e-8)
    cmap_fn = matplotlib.colormaps[_CMAP_MAP.get(cmap_name, 'viridis')]
    rgba    = (cmap_fn(U_norm) * 255).astype(np.uint8)
    buf     = io.BytesIO()
    Image.fromarray(rgba[..., :3]).save(buf, format='PNG')
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
st.set_page_config(page_title='CH4 Transitoire', layout='wide')
st.title('CH4 Transitoire — Surrogate Laplace AE')

st.sidebar.header('Modèle')
_model_names = list(MODELS.keys())
model_key = st.sidebar.radio('Pipeline', _model_names,
                              index=_model_names.index('Corrected'))

st.sidebar.markdown('---')
st.sidebar.header('Paramètres physiques θ')
slider_vals = []
for label, lo, hi, default, fmt in THETA_PARAMS:
    v = st.sidebar.slider(label, float(lo), float(hi), float(default),
                          step=float((hi - lo) / 200), format=fmt)
    slider_vals.append(v)
k_val, A_val, C_val = slider_vals

t_idx = st.sidebar.slider('t (instant)', 0, NT - 1, value=0)

st.sidebar.markdown('---')
_cmaps    = list(_CMAP_MAP.keys())
cmap_name = st.sidebar.selectbox('Colormap', _cmaps, index=0)

# ---------------------------------------------------------------------------
# Inférence + affichage
# ---------------------------------------------------------------------------
with st.spinner('Inférence en cours…'):
    U_pred = run_cached_inference(model_key, k_val, A_val, C_val, dt_val)

st.caption(
    f'θ = (k={k_val:.3f}, A={A_val:.1f}, C={C_val:.4f})  |  '
    f't = {t_idx}  (t·dt = {t_idx * dt_val:.2f} s)  |  '
    f'min `{U_pred[t_idx].min():.4f}`  max `{U_pred[t_idx].max():.4f}`'
)

st.image(render_frame(U_pred[t_idx], cmap_name), width=300)

# ---------------------------------------------------------------------------
# Courbe temporelle
# ---------------------------------------------------------------------------
with st.expander('Évolution temporelle (moyenne spatiale)', expanded=False):
    mean_t  = U_pred.mean(axis=(1, 2))
    fig_ts  = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=list(range(U_pred.shape[0])), y=mean_t.tolist(),
        mode='lines', line=dict(color='royalblue'),
    ))
    fig_ts.add_vline(x=t_idx, line_dash='dash', line_color='red')
    fig_ts.update_layout(
        xaxis_title='t', yaxis_title='mean(U)',
        height=240,
        margin=dict(l=10, r=10, t=10, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_ts, width='stretch')
