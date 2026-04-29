"""
transient/app.py — Visualisation Streamlit : theta -> champ CH4 transitoire U(t)
Usage : streamlit run transient/app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import base64
import io
import json
import numpy as np
import matplotlib
import torch
import streamlit as st
import streamlit.components.v1 as components
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


def _compute_upred(model_key: str, k: float, A: float, C: float, dt: float) -> np.ndarray:
    model, ckpt, device = get_model(model_key)
    theta = np.array([[k, A, C]], dtype=np.float32)
    return run_inference(theta, model, ckpt, device, dt=dt, k_max=CKPT_K_MAX)[0]  # (Nt, N, N)


def render_frame(frame: np.ndarray, cmap_name: str) -> bytes:
    vmin, vmax = 0.0, 0.05
    U_norm  = (frame - vmin) / max(vmax - vmin, 1e-8)
    cmap_fn = matplotlib.colormaps[_CMAP_MAP.get(cmap_name, 'viridis')]
    rgba    = (cmap_fn(U_norm) * 255).astype(np.uint8)
    buf     = io.BytesIO()
    Image.fromarray(rgba[..., :3]).save(buf, format='PNG', optimize=True)
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

st.sidebar.markdown('---')
t_idx = st.sidebar.slider('t (instant)', 0, NT - 1, value=0, key='t_slider')

st.sidebar.markdown('---')
_cmaps    = list(_CMAP_MAP.keys())
cmap_name = st.sidebar.selectbox('Colormap', _cmaps, index=0)

# ---------------------------------------------------------------------------
# Inférence  (U_pred mis en cache dans session_state)
# ---------------------------------------------------------------------------
_cache_key = (model_key, k_val, A_val, C_val, dt_val)
if st.session_state.get('upred_key') != _cache_key:
    with st.spinner('Inférence en cours…'):
        st.session_state['U_pred'] = _compute_upred(model_key, k_val, A_val, C_val, dt_val)
    st.session_state['upred_key'] = _cache_key

U_pred = st.session_state['U_pred']

# ---------------------------------------------------------------------------
# Pré-rendu des frames  (recalculé si U_pred ou colormap change)
# ---------------------------------------------------------------------------
_frames_key = (_cache_key, cmap_name)
if st.session_state.get('frames_key') != _frames_key:
    with st.spinner('Rendu des frames…'):
        st.session_state['frames'] = [render_frame(U_pred[i], cmap_name) for i in range(NT)]
    st.session_state['frames_key'] = _frames_key

frames = st.session_state['frames']

# ---------------------------------------------------------------------------
# Composant animation JS (toutes les frames en base64, animation client-side)
# ---------------------------------------------------------------------------
st.caption(f'θ = (k={k_val:.3f}, A={A_val:.1f}, C={C_val:.4f})')

frames_b64 = json.dumps([base64.b64encode(f).decode() for f in frames])

anim_html = f"""
<style>
  body {{ margin:0; background:transparent; }}
  .ctrl {{
    display:flex; align-items:center; gap:10px;
    margin-top:10px;
  }}
  #playbtn {{
    background:#4a90d9; border:none; border-radius:50%;
    width:38px; height:38px; font-size:18px; cursor:pointer;
    color:#fff; flex-shrink:0; line-height:1;
    transition: background .15s;
  }}
  #playbtn:hover {{ background:#2d72b8; }}
  #tslider {{
    flex:1; accent-color:#4a90d9; height:4px; cursor:pointer;
  }}
  .mono {{
    font-family:monospace; font-size:13px; color:#ccc;
    min-width:52px; text-align:right;
  }}
  .fps-row {{
    display:flex; align-items:center; gap:6px;
    margin-top:6px; font-size:12px; color:#aaa;
  }}
  #fpsinput {{
    width:44px; background:#2a2a2a; border:1px solid #555;
    border-radius:4px; color:#eee; padding:2px 4px;
    font-size:12px; text-align:center;
  }}
</style>
<div style="display:flex;flex-direction:column;align-items:center;font-family:sans-serif;padding:4px 8px">
  <img id="anim-frame"
       src="data:image/png;base64,{base64.b64encode(frames[t_idx]).decode()}"
       style="width:300px;height:300px;object-fit:contain;border-radius:4px"/>
  <div class="ctrl" style="width:340px">
    <button id="playbtn" onclick="togglePlay()">▶</button>
    <input type="range" id="tslider" min="0" max="{NT-1}" value="{t_idx}"
           oninput="seek(+this.value)"/>
    <span id="tlabel" class="mono">t = {t_idx}</span>
  </div>
  <div class="fps-row">
    FPS
    <input type="number" id="fpsinput" value="15" min="1" max="60"
           oninput="setFps(+this.value)"/>
  </div>
</div>
<script>
(function() {{
  const frames = {frames_b64};
  const img    = document.getElementById('anim-frame');
  const slider = document.getElementById('tslider');
  const label  = document.getElementById('tlabel');
  const btn    = document.getElementById('playbtn');

  let idx     = {t_idx};
  let playing = false;
  let fps     = 15;
  let handle  = null;

  function render() {{
    img.src = 'data:image/png;base64,' + frames[idx];
    slider.value = idx;
    label.textContent = 't = ' + idx;
  }}

  function advance() {{
    idx = (idx + 1) % frames.length;
    render();
  }}

  window.togglePlay = function() {{
    playing = !playing;
    btn.textContent = playing ? '⏸' : '▶';
    if (playing) {{
      handle = setInterval(advance, 1000 / fps);
    }} else {{
      clearInterval(handle);
    }}
  }};

  window.seek = function(i) {{
    idx = i;
    render();
  }};

  window.setFps = function(f) {{
    fps = f || 1;
    if (playing) {{
      clearInterval(handle);
      handle = setInterval(advance, 1000 / fps);
    }}
  }};

  render();
}})();
</script>
"""

components.html(anim_html, height=420, scrolling=False)

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
    fig_ts.update_layout(
        xaxis_title='t', yaxis_title='mean(U)',
        height=240,
        margin=dict(l=10, r=10, t=10, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_ts, width='stretch')
