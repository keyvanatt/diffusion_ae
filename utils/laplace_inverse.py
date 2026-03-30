import numpy as np
from scipy.optimize import nnls


def laplace_inverse(M, dt, s, Nt, E=None, rule='rect',
                    Lambda=0.0, LambdaSmooth=0.0,
                    NonNegative=False, ScaleColumns=True,
                    FinalSlopeWeight=0.0, FinalCurvatureWeight=0.0,
                    FinalWindow=5):
    """
    Reconstruction spatio-temporelle robuste depuis des modes de Laplace.

    Parametres
    ----------
    M    : ndarray (Nnodes, Ns)
    dt   : float > 0
    s    : ndarray (Ns,)
    Nt   : int – nombre de pas de temps a reconstruire
    E    : ndarray (Nt, Ns) ou None – matrice d'integration (recommande: meta['E'])
    rule : 'rect' ou 'trap'
    Lambda       : float >= 0 – regularisation Tikhonov
    LambdaSmooth : float >= 0 – lissage 1ere difference
    NonNegative  : bool – contrainte de positivite
    ScaleColumns : bool – normalise les colonnes d'E
    FinalSlopeWeight     : float >= 0
    FinalCurvatureWeight : float >= 0
    FinalWindow  : int >= 1

    Retour
    ------
    Crec : ndarray (Nnodes, Nt)
    info : dict
    """
    s = np.asarray(s).ravel()
    Nnodes, Ns = M.shape
    M = M.copy()

    # --- Construire / adopter E ---
    if E is None:
        t = np.arange(Nt) * dt
        w = np.ones(Nt)
        if rule == 'trap':
            w[0] = 0.5
            w[-1] = 0.5
        E = (dt * w)[:, None] * np.exp(-t[:, None] * s[None, :])
    else:
        E = E.copy()
        NtE, NsE = E.shape
        if NsE != Ns:
            raise ValueError(f"size(E,2)={NsE} incompatible avec Ns={Ns}.")
        if NtE != Nt:
            print(f"Warning: Nt demande={Nt}, size(E,1)={NtE}. Adoption de size(E,1).")
            Nt = NtE

    # --- Mise a l'echelle des colonnes d'E ---
    scaled = False
    if ScaleColumns:
        cn = np.maximum(np.linalg.norm(E, axis=0), np.finfo(float).eps)
        S_diag = 1.0 / cn
        E = E * S_diag[None, :]
        M = M * S_diag[None, :]
        scaled = True

    # --- Operateurs ---
    A = E.T  # (Ns, Nt)
    I = np.eye(Nt)

    # Differences premieres
    if LambdaSmooth > 0:
        D = np.zeros((Nt - 1, Nt))
        for k in range(Nt - 1):
            D[k, k] = -1.0
            D[k, k + 1] = 1.0
    else:
        D = np.zeros((0, Nt))

    # Penalisation locale de fin
    beta = FinalSlopeWeight
    beta2 = FinalCurvatureWeight
    Ktail = min(max(1, round(FinalWindow)), max(1, Nt - 1))
    useTail = (beta > 0) or (beta2 > 0)

    if useTail:
        Gs_list = []
        bs_list = []
        if beta > 0:
            for k in range(Ktail):
                g = np.zeros(Nt)
                g[-(k + 2)] = -1.0
                g[-(k + 1)] = 1.0
                Gs_list.append(np.sqrt(beta) * g)
                bs_list.append(0.0)
        if beta2 > 0 and Nt >= 3:
            K2 = min(2, Nt - 3)
            for k in range(K2 + 1):
                g2 = np.zeros(Nt)
                idx = Nt - k - 3
                g2[idx:idx + 3] = [1.0, -2.0, 1.0]
                Gs_list.append(np.sqrt(beta2) * g2)
                bs_list.append(0.0)
        if Gs_list:
            Gtail = np.vstack(Gs_list)
            btail = np.array(bs_list)
        else:
            Gtail = np.zeros((0, Nt))
            btail = np.zeros(0)
    else:
        Gtail = np.zeros((0, Nt))
        btail = np.zeros(0)

    # --- Resolution ligne par ligne ---
    Crec = np.zeros((Nnodes, Nt))
    res_sq_sum = 0.0

    for i in range(Nnodes):
        b = M[i, :]  # (Ns,)

        # Empilement augmente
        Aaug = A.copy()
        baug = b.copy()

        if Lambda > 0:
            Aaug = np.vstack([Aaug, np.sqrt(Lambda) * I])
            baug = np.concatenate([baug, np.zeros(Nt)])
        if LambdaSmooth > 0:
            Aaug = np.vstack([Aaug, np.sqrt(LambdaSmooth) * D])
            baug = np.concatenate([baug, np.zeros(D.shape[0])])
        if useTail:
            Aaug = np.vstack([Aaug, Gtail])
            baug = np.concatenate([baug, btail])

        # Solveur
        if NonNegative:
            c, _ = nnls(Aaug, baug)
        else:
            c, _, _, _ = np.linalg.lstsq(Aaug, baug, rcond=None)

        Crec[i, :] = c
        res_sq_sum += np.linalg.norm(Aaug @ c - baug) ** 2

    # --- Diagnostics ---
    data_resnorm = np.linalg.norm(Crec @ E - M, 'fro')
    data_relmisfit = data_resnorm / max(np.linalg.norm(M, 'fro'), np.finfo(float).eps)

    info = {
        'resnorm_total': np.sqrt(res_sq_sum),
        'data_resnorm': data_resnorm,
        'data_relmisfit': data_relmisfit,
        'rankE': np.linalg.matrix_rank(E),
        'lambda': Lambda,
        'mu': LambdaSmooth,
        'beta': beta,
        'beta2': beta2,
        'Ktail': Ktail,
        'scaled': scaled,
    }

    return Crec, info
