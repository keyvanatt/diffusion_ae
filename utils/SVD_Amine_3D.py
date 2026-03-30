import numpy as np


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

        for itloc in range(100):
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

            if err < 1e-6:
                break

        fact = (np.linalg.norm(R) * np.linalg.norm(S) * np.linalg.norm(T)) ** (1 / 3)
        alph.append(fact ** 3)
        F.append((R * fact / np.linalg.norm(R)).ravel())
        G.append((S * fact / np.linalg.norm(S)).ravel())
        P.append((T * fact / np.linalg.norm(T)).ravel())

        HH = HH - np.kron(T, np.kron(S, R)).reshape(nr, ns, nt, order='F')

        error_l2 = np.sqrt(np.sum(HH ** 2)) / error_l2_ini
        print(f"Err L2 = {error_l2:.6e}")
        Hist_ErrL2.append(error_l2)

        if np.isnan(fact) or fact ** 3 / alph[0] < erreur:
            break

    F = np.column_stack(F) if F else np.empty((nr, 0))
    G = np.column_stack(G) if G else np.empty((ns, 0))
    P = np.column_stack(P) if P else np.empty((nt, 0))
    alph = np.array(alph)

    return F, G, P, alph, Hist_ErrL2
