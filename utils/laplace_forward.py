import numpy as np


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
