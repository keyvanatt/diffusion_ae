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
              x0: np.ndarray) -> dolfinx.fem.Function:
        # Mise à jour des constantes
        self.D_c.value    = D
        self.b_c.value[:] = b_val

        # Réassemblage de A
        self.A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(self.A, self.a_form, bcs=[self.bc])
        self.A.assemble()

        # Réassemblage du RHS
        with self.b_rhs.localForm() as loc:
            loc.set(0)
        dolfinx.fem.petsc.assemble_vector(self.b_rhs, self.L_form)
        dolfinx.fem.petsc.apply_lifting(self.b_rhs, [self.a_form], bcs=[[self.bc]])
        self.b_rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        add_point_source(self.b_rhs, self.V, x0, f)
        dolfinx.fem.petsc.set_bc(self.b_rhs, [self.bc])

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
):
    """Wrapper one-shot — pour les appels ponctuels (benchmark, tests).
    Pour générer un dataset, préférer ConvDiffSimulator."""
    return ConvDiffSimulator(n=n, use_supg=use_supg).solve(D, b_val, f, x0)

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

def plot_sol(u_sol):
    """
    Affiche la solution u_sol à l'aide de Matplotlib.

    u_sol : fonction FEM
    """


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
    ax.legend(loc='upper right', fontsize=8)

    # Courbes de niveau
    ax = axes[1]
    ct = ax.tricontour(triang, u_sol.x.array.real, levels=15, cmap="RdBu_r")
    plt.colorbar(ct, ax=ax, label="u")
    ax.set_title("Isolignes")
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig("plots/convdiff_solution.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Figure sauvegardée → plots/convdiff_solution.png")

if __name__ == "__main__":
    u_sol = simulate()
    plot_sol(u_sol)
    grid = to_grid(u_sol, N_out=64)
    plt.imsave("plots/convdiff_grid.png", grid, cmap="hot", vmin=0.0, vmax=grid.max())
    print("Grille sauvegardée → plots/convdiff_grid.png")


