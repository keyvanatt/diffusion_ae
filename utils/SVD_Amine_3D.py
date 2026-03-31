import numpy as np


def svd_inverse_3d(F, G, P, alph):
    """
    Reconstruit un tenseur 3D a partir de la decomposition Tucker rang-1.

    Parametres
    ----------
    F    : ndarray (nr, nf_eff)
    G    : ndarray (ns, nf_eff)
    P    : ndarray (nt, nf_eff)
    alph : ndarray (nf_eff,)

    Retour
    ------
    HH_rec : ndarray (nr, ns, nt)
    """
    nr = F.shape[0]
    ns = G.shape[0]
    nt = P.shape[0]
    nf_eff = len(alph)

    HH_rec = np.zeros((nr, ns, nt))
    for k in range(nf_eff):
        fi = F[:, k:k+1]  # (nr, 1)
        gi = G[:, k:k+1]  # (ns, 1)
        pi = P[:, k:k+1]  # (nt, 1)
        HH_rec += alph[k] * np.kron(pi, np.kron(gi, fi)).reshape(nr, ns, nt, order='F')

    return HH_rec


def svd_amine_3d(HH, nf, erreur):
    """
    Decomposition Tucker rang-1 iterative d'un tenseur 3D.

    Parametres
    ----------
    HH : ndarray (nr, ns, nt)
    nf : int   – nombre max de modes
    erreur : float – seuil relatif d'arret

    Retour
    ------
    F : ndarray (nr, nf_eff)
    G : ndarray (ns, nf_eff)
    P : ndarray (nt, nf_eff)
    alph : ndarray (nf_eff,)
    Hist_ErrL2 : list[float]
    """
    error_l2_ini = np.sqrt(np.sum(HH ** 2))

    Hist_ErrL2 = [1.0]

    HH = HH.copy()
    nr, ns, nt = HH.shape
    alph = []
    F = []
    G = []
    P = []

    for itglob in range(nf):
        R = np.random.rand(nr, 1)
        S = np.random.rand(ns, 1)
        T = np.random.rand(nt, 1)

        Hrst = HH
        Hstr = np.transpose(Hrst, (1, 2, 0))  # permute [2 3 1]
        Htrs = np.transpose(Hrst, (2, 0, 1))  # permute [3 1 2]

        for itloc in range(500):
            RST = np.kron(T, np.kron(S, R)).reshape(nr, ns, nt, order='F')

            # R update
            R = (Hrst.reshape(nr, ns * nt, order='F') @ np.kron(T, S)) / (
                (T.T @ T) * (S.T @ S)
            )
            # S update
            S = (Hstr.reshape(ns, nt * nr, order='F') @ np.kron(R, T)) / (
                (R.T @ R) * (T.T @ T)
            )
            # T update
            T = (Htrs.reshape(nt, nr * ns, order='F') @ np.kron(S, R)) / (
                (S.T @ S) * (R.T @ R)
            )

            RST_new = np.kron(T, np.kron(S, R)).reshape(nr, ns, nt, order='F')
            err = np.sqrt(np.sum((RST - RST_new) ** 2))

            if err < 1e-8:
                break

        nR, nS, nT = np.linalg.norm(R), np.linalg.norm(S), np.linalg.norm(T)
        amplitude = nR * nS * nT
        alph.append(amplitude)
        F.append((R / nR).ravel())
        G.append((S / nS).ravel())
        P.append((T / nT).ravel())

        HH = HH - np.kron(T, np.kron(S, R)).reshape(nr, ns, nt, order='F')

        error_l2 = np.sqrt(np.sum(HH ** 2)) / error_l2_ini
        print(f"Err L2 = {error_l2:.6e}")
        Hist_ErrL2.append(error_l2)

        if np.isnan(amplitude) or amplitude / alph[0] < erreur:
            break

    F = np.column_stack(F) if F else np.empty((nr, 0))
    G = np.column_stack(G) if G else np.empty((ns, 0))
    P = np.column_stack(P) if P else np.empty((nt, 0))
    alph = np.array(alph)

    return F, G, P, alph, Hist_ErrL2
