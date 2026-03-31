import numpy as np


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
