import numpy as np
import torch


def laplace_forward(C, dt, rule='trap', gamma=0.0):
    """
    Transformee de Laplace d'un champ spatio-temporel discret.
    Utilise des s complexes sur le contour de Bromwich (FFT).

    Parametres
    ----------
    C     : ndarray (Nnodes, Nt)
    dt    : float > 0
    rule  : 'rect' ou 'trap' – regle de quadrature temporelle
    gamma : float >= 0 – amortissement (0 = DFT pure)

    Retour
    ------
    M    : ndarray (Nnodes, Nt) complex
    s    : ndarray (Nt,) complex  –  s_k = gamma + i*omega_k
    meta : dict
    """
    Nnodes, Nt = C.shape
    t = np.arange(Nt) * dt

    w = np.ones(Nt)
    if rule == 'trap':
        w[0] = 0.5
        w[-1] = 0.5

    a = C * (dt * w * np.exp(-gamma * t))[None, :]
    M = np.fft.fft(a, axis=1)

    omega = 2 * np.pi * np.fft.fftfreq(Nt, d=dt)
    s = gamma + 1j * omega

    meta = {'rule': rule, 'dt': dt, 'Nt': Nt, 'gamma': gamma, 'w': w}
    return M, s, meta



def laplace_inverse(M, dt, Nt, rule='trap', gamma=0.0):
    """
    Inversion de la transformee de Laplace par IFFT (contour de Bromwich).

    Parametres
    ----------
    M     : ndarray (Nnodes, Nt) complex  – retourne par laplace_forward
    dt    : float > 0
    Nt    : int
    rule  : 'rect' ou 'trap'  – doit correspondre au forward
    gamma : float >= 0        – doit correspondre au forward

    Retour
    ------
    Crec : ndarray (Nnodes, Nt) reel
    info : dict
    """
    t = np.arange(Nt) * dt
    w = np.ones(Nt)
    if rule == 'trap':
        w[0] = 0.5
        w[-1] = 0.5

    a_rec = np.fft.ifft(M, n=Nt, axis=1)
    Crec = np.real(a_rec) / (dt * w * np.exp(-gamma * t))[None, :]

    rel_imag = np.linalg.norm(np.imag(a_rec)) / max(np.linalg.norm(a_rec), 1e-300)
    return Crec, {'residual_imag_ratio': rel_imag}


def laplace_forward_tik(C, s_list, dt, rule='trap'):
    """
    Forward Laplace at arbitrary s points.  Differentiable (PyTorch).

    Computes  U_hat[n, k] = sum_t C[n,t] * dt * w[t] * exp(-s_k * t)

    Parameters
    ----------
    C      : (Nnodes, Nt) real  – ndarray or Tensor
    s_list : (K,) complex       – ndarray or Tensor
    dt     : float
    rule   : 'rect' | 'trap'

    Returns
    -------
    U_hat : (Nnodes, K) complex Tensor
    """
    if not isinstance(C, torch.Tensor):
        C = torch.tensor(C, dtype=torch.float64)
    if not isinstance(s_list, torch.Tensor):
        s_list = torch.tensor(s_list, dtype=torch.complex128)

    device = s_list.device
    C = C.to(device=device)
    Nnodes, Nt = C.shape
    t = torch.arange(Nt, dtype=torch.float64, device=device) * dt

    w = torch.ones(Nt, dtype=torch.float64, device=device)
    if rule == 'trap':
        w[0] = 0.5
        w[-1] = 0.5

    # F[k, t] = dt * w[t] * exp(-s_k * t)   (K, Nt) complex
    F = dt * w[None, :] * torch.exp(-s_list[:, None] * t[None, :])
    return C.to(dtype=F.dtype) @ F.T   # (Nnodes, K)


def laplace_inverse_tik(U_hat, s_list, dt, Nt, alpha_t, lam, rule='trap'):
    """
    Regularised Laplace inversion at arbitrary s points.  Differentiable (PyTorch).

    Solves the normal equations with Tikhonov regularization:
        A v* = Re(F^H û)
        A   = Re(F^H F) + alpha_t * D_t^T D_t + lam * I

    Conjugate symmetry: frequencies with Im(s) > 0 are extended by their conjugates
    before forming the normal equations (doubles effective constraints cheaply).

    Works in float32 or float64 depending on s_list.dtype:
      - complex128 (default) → float64, CPU, best accuracy
      - complex64            → float32, stays on s_list.device, GPU-differentiable

    Parameters
    ----------
    U_hat  : (Nnodes, K) complex  – output of laplace_forward_tik
    s_list : (K,) complex Tensor
    dt     : float
    Nt     : int   – number of time steps to reconstruct
    alpha_t: float – temporal-smoothness weight
    lam    : float – ridge weight (use > 0 when K < Nt to avoid singularity)
    rule   : 'rect' | 'trap'  – must match forward

    Returns
    -------
    V_rec : (Nnodes, Nt) real Tensor  (same device as s_list)
    """
    if not isinstance(U_hat, torch.Tensor):
        U_hat = torch.tensor(U_hat, dtype=torch.complex128)
    if not isinstance(s_list, torch.Tensor):
        s_list = torch.tensor(s_list, dtype=torch.complex128)

    device = s_list.device
    cdtype = s_list.dtype                                    # complex64 or complex128
    rdtype = torch.float32 if cdtype == torch.complex64 else torch.float64

    U_hat = U_hat.to(device=device, dtype=cdtype)
    t = torch.arange(Nt, dtype=rdtype, device=device) * dt
    w = torch.ones(Nt, dtype=rdtype, device=device)
    if rule == 'trap':
        w[0] = 0.5; w[-1] = 0.5

    # Conjugate extension: Im(s) > 0 → add conjugate pair
    c_mask     = s_list.imag > 0
    s_full     = torch.cat([s_list, torch.conj(s_list[c_mask])])
    U_hat_full = torch.cat([U_hat, torch.conj(U_hat[:, c_mask])], dim=1)  # (Nnodes, K_full)

    # F_full[k, t] = dt * w[t] * exp(-s_full[k] * t)   (K_full, Nt)
    F_full = dt * w[None, :] * torch.exp(-s_full[:, None] * t[None, :])
    FH     = torch.conj(F_full).T                           # (Nt, K_full)
    FtF    = torch.real(FH @ F_full)                        # (Nt, Nt) rdtype

    Dt    = (torch.diag(torch.ones(Nt - 1, dtype=rdtype, device=device), 1)
             - torch.eye(Nt, dtype=rdtype, device=device))[:Nt - 1, :]
    DtTDt = Dt.T @ Dt

    A   = FtF + alpha_t * DtTDt + lam * torch.eye(Nt, dtype=rdtype, device=device)
    RHS = torch.real(U_hat_full @ torch.conj(F_full))       # (Nnodes, Nt)

    return torch.linalg.solve(A, RHS.T).T                   # (Nnodes, Nt) rdtype
