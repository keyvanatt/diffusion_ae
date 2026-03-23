"""
app.py - Visualisation Streamlit : theta -> grille U predite par le CVAE
Usage : streamlit run app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from main import load_model

THETA_PARAMS = [
    ('D   - diffusivite',         0.001,  0.10,  0.02),
    ('|b| - intensite advection',  0.0,    2.83,  0.0),
    ('angle b (deg)',              0.0,  360.0,   0.0),
    ('x0x - source centre x',      0.1,    0.9,   0.5),
    ('x0y - source centre y',      0.1,    0.9,   0.5),
    ('f   - intensite source',     1.0,   20.0,  10.0),
]
DATASET_PATH = 'dataset/dataset.npz'
CKPT         = 'checkpoints/cvae_best.pt'


@st.cache_resource
def get_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, ckpt = load_model(CKPT, device)
    return model, ckpt, device


@st.cache_data
def get_dataset():
    d = np.load(DATASET_PATH, allow_pickle=True)
    return d['U'].astype(np.float32), d['theta'].astype(np.float32)


@torch.no_grad()
def run_predict(theta_raw, model, ckpt, device):
    theta_mean = ckpt['theta_mean'].to(device)
    theta_std  = ckpt['theta_std'].to(device)
    U_min      = float(ckpt['U_min'])
    U_max      = float(ckpt['U_max'])
    theta_t    = torch.tensor(theta_raw, dtype=torch.float32, device=device)
    theta_norm = (theta_t - theta_mean) / theta_std
    U_hat_norm = model.generate(theta_norm, n_samples=1)
    U_pred     = (U_hat_norm + 1.0) / 2.0 * (U_max - U_min) + U_min
    return U_pred[0, 0].cpu().numpy()


def find_nearest(theta_raw, theta_all):
    theta_raw = np.array(theta_raw, dtype=np.float32)
    std  = theta_all.std(axis=0) + 1e-8
    diff = (theta_all - theta_raw) / std
    return int(np.argmin((diff ** 2).sum(axis=1)))


def make_heatmap_fig(grids, cmap):
    titles = list(grids.keys())
    n      = len(titles)
    fig    = make_subplots(rows=n, cols=1, subplot_titles=titles,
                           vertical_spacing=0.06)

    arrays = list(grids.values())
    shared = np.concatenate([a.ravel() for a in arrays[:2]])
    zmin_s, zmax_s = float(shared.min()), float(shared.max())

    for i, (title, arr) in enumerate(grids.items(), start=1):
        is_err     = 'Erreur' in title
        colorscale = 'Reds' if is_err else cmap
        zmin = float(arr.min()) if is_err else zmin_s
        zmax = float(arr.max()) if is_err else zmax_s
        fig.add_trace(
            go.Heatmap(
                z=arr,
                colorscale=colorscale,
                zmin=zmin, zmax=zmax,
                showscale=True,
                colorbar=dict(len=0.3, thickness=12, y=1 - (i-1)/n - 0.15,
                              tickfont=dict(size=10)),
            ),
            row=i, col=1,
        )

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        height=320 * n,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


# ── Page ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='CVAE Diffusion', layout='wide')
st.title('CVAE — Diffusion-Advection')

st.sidebar.header('Parametres physiques theta')
slider_vals = []
for label, lo, hi, default in THETA_PARAMS:
    v = st.sidebar.slider(label, float(lo), float(hi), float(default),
                          step=float((hi - lo) / 200))
    slider_vals.append(v)

# Convertir (|b|, angle) -> (bx, by)
D, b_mag, b_angle_deg, x0x, x0y, f = slider_vals
b_rad = np.deg2rad(b_angle_deg)
bx    = b_mag * np.cos(b_rad)
by    = b_mag * np.sin(b_rad)
theta_vals = [D, bx, by, x0x, x0y, f]

st.sidebar.markdown('---')
cmap_name = st.sidebar.selectbox('Colormap',
    ['Viridis', 'Plasma', 'Inferno', 'RdBu', 'Turbo'], index=0)
gt_btn = st.sidebar.button('Comparer a la ground truth', use_container_width=True)

# Inference live — re-run automatique a chaque slider
model, ckpt, device = get_model()
U_pred = run_predict(theta_vals, model, ckpt, device)

# Memoriser l'etat GT entre les re-runs
if gt_btn:
    st.session_state['show_gt'] = True
    st.session_state['gt_theta'] = theta_vals[:]   # theta au moment du clic

st.caption(f'bx={bx:.3f}  by={by:.3f}  |  min `{U_pred.min():.3f}` max `{U_pred.max():.3f}` mean `{U_pred.mean():.3f}`')

if st.session_state.get('show_gt'):
    if st.button('Effacer GT'):
        del st.session_state['show_gt']
        del st.session_state['gt_theta']
        st.rerun()

if True:
    if st.session_state.get('show_gt'):
        U_all, theta_all = get_dataset()
        idx  = find_nearest(st.session_state['gt_theta'], theta_all)
        U_gt = U_all[idx]
        err  = np.abs(U_pred - U_gt)
        rmse = float(np.sqrt(np.mean(err ** 2)))
        grids = {
            f'Ground truth (idx {idx})': U_gt,
            'Prediction CVAE':           U_pred,
            f'|Erreur| RMSE={rmse:.4f}': err,
        }
        st.caption(
            f'Voisin le plus proche : idx **{idx}** — '
            f'theta = {[f"{v:.3f}" for v in theta_all[idx].tolist()]}'
        )
    else:
        grids = {'Prediction CVAE': U_pred}

    st.plotly_chart(make_heatmap_fig(grids, cmap_name), use_container_width=True)
