import numpy as np


def laplace_forward(C, dt, s=None, rule='rect', Ns=7, sMin=None, sMax=None):
    """
    Transformee de Laplace tronquee d'un champ spatio-temporel discret.

    Parametres
    ----------
    C    : ndarray (Nnodes, Nt)  – champ spatio-temporel
    dt   : float > 0             – pas de temps
    s    : ndarray (Ns,) ou None – liste de s >= 0 (auto si None)
    rule : 'rect' ou 'trap'      – regle de quadrature
    Ns   : int                   – nombre de points s auto (ignore si s fourni)
    sMin : float ou None
    sMax : float ou None

    Retour
    ------
    M    : ndarray (Nnodes, Ns)
    s    : ndarray (Ns,)
    meta : dict avec t, E, rule, dt, Nt, Ns
    """
    Nnodes, Nt = C.shape
    t = np.arange(Nt) * dt
    T = t[-1] if Nt > 1 else 0.0

    # --- Choix de s ---
    if s is None or len(s) == 0:
        if sMin is None:
            sMin = 1.0 / max(T, np.finfo(float).eps)
        if sMax is None:
            sMax = 10.0 / dt
        s = np.logspace(np.log10(sMin), np.log10(sMax), Ns)
    else:
        s = np.asarray(s).ravel()
        Ns = len(s)

    # --- Poids temporels ---
    w = np.ones(Nt)
    if rule == 'trap':
        w[0] = 0.5
        w[-1] = 0.5

    # --- Matrice exponentielle E(n,j) = dt * w_n * exp(-s_j * t_n) ---
    E = (dt * w)[:, None] * np.exp(-t[:, None] * s[None, :])  # (Nt, Ns)

    # --- Transformee ---
    M = C @ E  # (Nnodes, Ns)

    meta = {
        't': t,
        'E': E,
        'rule': rule,
        'dt': dt,
        'Nt': Nt,
        'Ns': Ns,
    }

    return M, s, meta
