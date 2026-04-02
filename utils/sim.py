import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.mesh
import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.geometry
import dolfinx.io

import ufl
from ufl import dx, grad, dot, inner, TestFunction, TrialFunction
from ufl import CellDiameter, sqrt, conditional, gt
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
import matplotlib.tri as mtri


print(f"DOLFINx version : {dolfinx.__version__}") # pyright: ignore[reportPrivateImportUsage]

def _obstacle_bc(V, rect):
    """
    Retourne un DirichletBC u=0 sur les dofs à l'intérieur du rectangle.

    rect : (x0, y0, x1, y1) avec x0 < x1 et y0 < y1, en coordonnées du domaine [0,1]²
    Retourne None si rect est None.
    """
    if rect is None:
        return None
    rx0, ry0, rx1, ry1 = rect
    assert rx0 < rx1 and ry0 < ry1, \
        f"Rectangle invalide : rect={rect} — vérifier que x0<x1 et y0<y1"
    obs_dofs = dolfinx.fem.locate_dofs_geometrical(
        V, lambda x: (x[0] >= rx0) & (x[0] <= rx1) & (x[1] >= ry0) & (x[1] <= ry1)
    )
    return dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0), obs_dofs, V)


def add_point_source(b_vec, V, x0_2d, magnitude):
    """
    Injecte f·δ(x−x0) dans le vecteur RHS.

    Principe : L(v) = f·v(x0)
    Pour P1, v(x0) = Σ_i φ_i(x0) * v_i
    où φ_i(x0) = coordonnée barycentrique i de x0 dans son triangle.
    On injecte donc f·φ_i(x0) dans la composante i du RHS.
    """
    x0_3d = np.array([[x0_2d[0], x0_2d[1], 0.0]])

    # Trouver le triangle contenant x0
    bb_tree    = dolfinx.geometry.bb_tree(V.mesh, V.mesh.topology.dim)
    candidates = dolfinx.geometry.compute_collisions_points(bb_tree, x0_3d)
    cells      = dolfinx.geometry.compute_colliding_cells(V.mesh, candidates, x0_3d)

    if len(cells.links(0)) == 0:
        raise ValueError(f"x0={x0_2d} est hors du domaine !")

    cell_id = cells.links(0)[0]

    # Coordonnées des 3 sommets du triangle
    verts_idx = V.mesh.geometry.dofmap[cell_id]
    verts     = V.mesh.geometry.x[verts_idx][:, :2]

    # Coordonnées barycentriques : φ_i(x0)
    T    = verts[:2, :] - verts[2, :]
    rhs  = x0_2d - verts[2, :]
    lam  = np.linalg.solve(T.T, rhs)
    bary = np.array([lam[0], lam[1], 1.0 - lam[0] - lam[1]])

    # Injecter dans le RHS
    dofs = V.dofmap.cell_dofs(cell_id)
    for i, dof in enumerate(dofs):
        b_vec.setValue(int(dof), magnitude * bary[i], True)

    b_vec.assemble()


class ConvDiffSimulator:
    """
    Compile les formes FEM une seule fois à la construction, puis résout
    pour des paramètres (D, b, f, x0) différents sans recompilation JIT.

    Usage
    -----
    sim = ConvDiffSimulator(n=64)
    u_sol = sim.solve(D=0.01, b_val=np.array([1.0, 0.3]), f=10.0, x0=np.array([0.5, 0.5]))
    """

    def __init__(self, n: int = 64, use_supg: bool = True):
        self.msh = dolfinx.mesh.create_unit_square(
            MPI.COMM_WORLD, n, n,
            cell_type=dolfinx.mesh.CellType.triangle
        )
        self.V = dolfinx.fem.functionspace(self.msh, ("Lagrange", 1))

        # Constantes mutables — pas de recompilation quand on change leur valeur
        self.D_c = dolfinx.fem.Constant(self.msh, PETSc.ScalarType(1e-2))
        self.b_c = dolfinx.fem.Constant(
            self.msh, np.array([1.0, 0.0], dtype=float)
        )
        zero_c = dolfinx.fem.Constant(self.msh, PETSc.ScalarType(0.0))

        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        a_std = (self.D_c * dot(grad(u), grad(v)) * dx
                 + dot(self.b_c, grad(u)) * v * dx)  # type: ignore

        if use_supg:
            h      = CellDiameter(self.msh)
            b_norm = sqrt(dot(self.b_c, self.b_c))
            tau    = conditional(gt(b_norm, 1e-10), h / (2 * b_norm), 0.0)
            a_supg = tau * dot(self.b_c, grad(v)) * dot(self.b_c, grad(u)) * dx
        else:
            a_supg = 0.0

        a = a_std + a_supg
        L = zero_c * v * dx

        # Dirichlet u=0 sur tout le bord (fixe)
        self.msh.topology.create_connectivity(
            self.msh.topology.dim - 1, self.msh.topology.dim
        )
        boundary_facets = dolfinx.mesh.exterior_facet_indices(self.msh.topology)
        boundary_dofs   = dolfinx.fem.locate_dofs_topological(
            self.V, self.msh.topology.dim - 1, boundary_facets
        )
        self.bc = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, self.V)

        # Compilation JIT — une seule fois
        self.a_form = dolfinx.fem.form(a)
        self.L_form = dolfinx.fem.form(L)

        # Pré-allocation
        self.A     = dolfinx.fem.petsc.create_matrix(self.a_form)
        self.b_rhs = dolfinx.fem.petsc.create_vector(self.L_form)

        # Solveur KSP réutilisé
        self.ksp = PETSc.KSP().create(self.msh.comm)
        self.ksp.setType("gmres")
        self.ksp.getPC().setType("ilu")
        self.ksp.setTolerances(rtol=1e-10)
        self.ksp.setFromOptions()

    def solve(self, D: float, b_val: np.ndarray, f: float,
              x0: np.ndarray, rect=None) -> dolfinx.fem.Function:
        if rect is not None:
            assert not (rect[0] <= x0[0] <= rect[2] and rect[1] <= x0[1] <= rect[3]), \
                f"La source x0={x0} est à l'intérieur de l'obstacle rect={rect}"
        bcs = [self.bc]
        obs_bc = _obstacle_bc(self.V, rect)
        if obs_bc is not None:
            bcs.append(obs_bc)

        # Mise à jour des constantes
        self.D_c.value    = D
        self.b_c.value[:] = b_val

        # Réassemblage de A
        self.A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(self.A, self.a_form, bcs=bcs)
        self.A.assemble()

        # Réassemblage du RHS
        with self.b_rhs.localForm() as loc:
            loc.set(0)
        dolfinx.fem.petsc.assemble_vector(self.b_rhs, self.L_form)
        dolfinx.fem.petsc.apply_lifting(self.b_rhs, [self.a_form], bcs=[bcs])
        self.b_rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        add_point_source(self.b_rhs, self.V, x0, f)
        dolfinx.fem.petsc.set_bc(self.b_rhs, bcs)

        u_sol = dolfinx.fem.Function(self.V)
        self.ksp.setOperators(self.A)
        self.ksp.solve(self.b_rhs, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        return u_sol


def simulate(
    D=1e-2,
    b_val=np.array([1.0, 0.3]),
    f=10.0,
    x0=np.array([0.5, 0.5]),
    n=64,
    use_supg=True,
    rect=None,
):
    """Wrapper one-shot — pour les appels ponctuels (benchmark, tests).
    Pour générer un dataset, préférer ConvDiffSimulator."""
    return ConvDiffSimulator(n=n, use_supg=use_supg).solve(D, b_val, f, x0, rect=rect)


class ConvDiffTransientSimulator:
    """
    Résout l'équation de convection-diffusion transitoire :

        ∂u/∂t + b·∇u − D·Δu = f·δ(x−x0),  t > 0
        u(x, 0) = 0   (source activée à t=0, conditions initiales nulles)
        u = 0 sur ∂Ω

    Schéma : Euler implicite (inconditionnellement stable).
    Stabilisation SUPG sur la partie convective.

    Usage
    -----
    sim = ConvDiffTransientSimulator(n=64, dt=0.01)
    u_list = sim.solve(D=0.01, b_val=np.array([1.0, 0.3]), f=10.0,
                       x0=np.array([0.5, 0.5]), n_steps=50)
    # u_list[k] est la solution FEM à t=(k+1)*dt
    """

    def __init__(self, n: int = 64, dt: float = 0.01, use_supg: bool = True):
        self.msh = dolfinx.mesh.create_unit_square(
            MPI.COMM_WORLD, n, n,
            cell_type=dolfinx.mesh.CellType.triangle
        )
        self.V = dolfinx.fem.functionspace(self.msh, ("Lagrange", 1))
        self.dt_val = dt

        # Constantes mutables
        self.D_c  = dolfinx.fem.Constant(self.msh, PETSc.ScalarType(1e-2))
        self.b_c  = dolfinx.fem.Constant(self.msh, np.array([1.0, 0.0], dtype=float))
        self.dt_c = dolfinx.fem.Constant(self.msh, PETSc.ScalarType(dt))

        u   = TrialFunction(self.V)
        v   = TestFunction(self.V)

        # u^n — solution au pas précédent, mise à jour à chaque itération
        self.u_n = dolfinx.fem.Function(self.V)

        # Forme bilinéaire : ∫ u v dx + dt [D ∫ ∇u·∇v dx + ∫ b·∇u v dx]
        a_std = (u * v * dx
                 + self.dt_c * self.D_c * dot(grad(u), grad(v)) * dx
                 + self.dt_c * dot(self.b_c, grad(u)) * v * dx)  # type: ignore

        if use_supg:
            h      = CellDiameter(self.msh)
            b_norm = sqrt(dot(self.b_c, self.b_c))
            tau    = conditional(gt(b_norm, 1e-10), h / (2 * b_norm), 0.0)
            # Stabilisation SUPG de la partie convective uniquement
            a_supg = self.dt_c * tau * dot(self.b_c, grad(v)) * dot(self.b_c, grad(u)) * dx
        else:
            a_supg = 0.0

        # Forme linéaire : ∫ u^n v dx  (source ponctuelle ajoutée manuellement)
        L = self.u_n * v * dx

        # Conditions de Dirichlet u=0 sur tout le bord
        self.msh.topology.create_connectivity(
            self.msh.topology.dim - 1, self.msh.topology.dim
        )
        boundary_facets = dolfinx.mesh.exterior_facet_indices(self.msh.topology)
        boundary_dofs   = dolfinx.fem.locate_dofs_topological(
            self.V, self.msh.topology.dim - 1, boundary_facets
        )
        self.bc = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, self.V)

        # Compilation JIT — une seule fois
        self.a_form = dolfinx.fem.form(a_std + a_supg)
        self.L_form = dolfinx.fem.form(L)

        # Pré-allocation
        self.A     = dolfinx.fem.petsc.create_matrix(self.a_form)
        self.b_rhs = dolfinx.fem.petsc.create_vector(self.L_form)

        # Solveur KSP — LU direct : factorisation une fois par sample,
        # substitutions avant/arrière pour chaque pas de temps (pas d'itérations)
        self.ksp = PETSc.KSP().create(self.msh.comm)
        self.ksp.setType("preonly")
        self.ksp.getPC().setType("lu")
        self.ksp.setFromOptions()

    def solve(
        self, D: float, b_val: np.ndarray, f: float,
        x0: np.ndarray, n_steps: int,
        tol: float = 1e-4,
        rect=None,
    ) -> list:
        """
        Résout jusqu'à n_steps pas de temps (pas = dt).
        Retourne la liste [u^1, ..., u^k] avec k ≤ n_steps.

        Arrêt anticipé dès que ||u^{n+1} − u^n|| / ||u^n|| < tol
        (régime stationnaire atteint). Mettre tol=0 pour désactiver.

        rect : (x0, y0, x1, y1) — obstacle rectangulaire, u=0 à l'intérieur.
        """
        if rect is not None:
            assert not (rect[0] <= x0[0] <= rect[2] and rect[1] <= x0[1] <= rect[3]), \
                f"La source x0={x0} est à l'intérieur de l'obstacle rect={rect}"
        bcs = [self.bc]
        obs_bc = _obstacle_bc(self.V, rect)
        if obs_bc is not None:
            bcs.append(obs_bc)

        self.D_c.value    = D
        self.b_c.value[:] = b_val

        # Condition initiale : u^0 = 0
        self.u_n.x.array[:] = 0.0

        # Matrice de gauche (ne dépend pas de t ni de u^n)
        self.A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(self.A, self.a_form, bcs=bcs)
        self.A.assemble()
        self.ksp.setOperators(self.A)

        solutions = []
        for _ in range(n_steps):
            with self.b_rhs.localForm() as loc:
                loc.set(0)
            dolfinx.fem.petsc.assemble_vector(self.b_rhs, self.L_form)
            add_point_source(self.b_rhs, self.V, x0, self.dt_val * f)
            dolfinx.fem.petsc.apply_lifting(self.b_rhs, [self.a_form], bcs=[bcs])
            self.b_rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            dolfinx.fem.petsc.set_bc(self.b_rhs, bcs)

            u_new = dolfinx.fem.Function(self.V)
            self.ksp.solve(self.b_rhs, u_new.x.petsc_vec)
            u_new.x.scatter_forward()

            solutions.append(u_new)

            if tol > 0:
                diff  = np.linalg.norm(u_new.x.array - self.u_n.x.array)
                scale = np.linalg.norm(self.u_n.x.array) + 1e-12
                if diff / scale < tol:
                    # Répéter le dernier pas pour garder la shape (n_steps, N, N)
                    solutions += [u_new] * (n_steps - len(solutions))
                    break

            self.u_n.x.array[:] = u_new.x.array

        return solutions


def simulate_transient(
    D=1e-2,
    b_val=np.array([1.0, 0.3]),
    f=10.0,
    x0=np.array([0.5, 0.5]),
    n=64,
    dt=0.01,
    n_steps=100,
    use_supg=True,
    rect=None,
):
    """Wrapper one-shot pour la simulation transitoire.
    Pour générer un dataset, préférer ConvDiffTransientSimulator."""
    return ConvDiffTransientSimulator(n=n, dt=dt, use_supg=use_supg).solve(
        D, b_val, f, x0, n_steps, rect=rect
    )


def to_grid_sequence(u_list, N_out=64) -> np.ndarray:
    """
    Convertit une liste de solutions FEM (retournée par ConvDiffTransientSimulator.solve)
    en tableau numpy de shape (n_steps, N_out, N_out).
    """
    frames = [to_grid(u, N_out) for u in u_list]
    return np.stack(frames, axis=0)

def to_grid(u_sol, N_out=64):
    """
    Version optimisée de fem_to_grid avec scipy — idéale pour générer
    un grand dataset car évite la boucle Python point par point.
 
    ~10× plus rapide que la version dolfinx pour N_out > 128.
    """
    msh = u_sol.function_space.mesh
    coords = msh.geometry.x[:, :2]
    u_fem  = u_sol.x.array.real
 
    xx  = np.linspace(0.0, 1.0, N_out)
    yy  = np.linspace(0.0, 1.0, N_out)
    XX, YY = np.meshgrid(xx, yy)
 
    U = griddata(coords, u_fem, (XX, YY), method='linear', fill_value=0.0)
    return U.astype(np.float32)

def plot_sol(u_sol, rect=None):
    """
    Affiche la solution u_sol à l'aide de Matplotlib.

    u_sol : fonction FEM
    rect  : (x0, y0, x1, y1) obstacle rectangulaire à superposer (optionnel)
    """
    import matplotlib.patches as mpatches

    msh = u_sol.function_space.mesh

    # Extraire les coordonnées et la triangulation
    coords = msh.geometry.x[:, :2]          # (n_nodes, 2)
    cells  = msh.geometry.dofmap            # connectivité des triangles
    triang = mtri.Triangulation(coords[:, 0], coords[:, 1], cells)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Carte de couleur
    ax = axes[0]
    cf = ax.tricontourf(triang, u_sol.x.array.real, levels=40, cmap="hot")
    plt.colorbar(cf, ax=ax, label="u")
    ax.set_title("Solution u colormap")
    ax.set_aspect('equal')

    # Courbes de niveau
    ax = axes[1]
    ct = ax.tricontour(triang, u_sol.x.array.real, levels=15, cmap="RdBu_r")
    plt.colorbar(ct, ax=ax, label="u")
    ax.set_title("Isolignes")
    ax.set_aspect('equal')

    if rect is not None:
        rx0, ry0, rx1, ry1 = rect
        for ax in axes:
            ax.add_patch(mpatches.Rectangle(
                (rx0, ry0), rx1 - rx0, ry1 - ry0,
                linewidth=1.5, edgecolor='white', facecolor='gray', alpha=0.7
            ))

    plt.tight_layout()
    plt.savefig("plots/convdiff_solution.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Figure sauvegardée → plots/convdiff_solution.png")

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from utils.animate import animate

    rect = (0.7, 0.45, 0.8, 0.55)   # obstacle rectangulaire (x0, y0, x1, y1)

    #stationaire
    u_sol = simulate(
        D=1e-2,
        b_val=np.array([2.0, 0.0]),
        f=10.0,
        rect=rect)
    plot_sol(u_sol, rect=rect)
    grid = to_grid(u_sol, N_out=64)
    plt.imsave("plots/convdiff_grid.png", grid, cmap="hot", vmin=0.0, vmax=grid.max())
    print("Grille sauvegardée → plots/convdiff_grid.png")

    #transitoire
    dt= 0.05
    u_list = simulate_transient(dt=dt,
                                n_steps=100,
                                D=1e-1,
                                b_val=np.array([2.0, 0.0]),
                                f=10,
                                rect=rect,
                                )
    grids  = to_grid_sequence(u_list, N_out=64)   # (n_steps, 64, 64)
    print(f"Séquence transitoire : shape={grids.shape}, max={grids.max():.4f}")

    animate(
        grids,
        output_path="plots/convdiff_transient.gif",
        fps=10,
        cmap="hot",
        label="u",
        title_fn=lambda t: f"t = {(t + 1) * dt:.3f}",
        rect=rect,
    )


