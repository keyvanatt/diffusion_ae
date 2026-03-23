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


def simulate(
    D=1e-2, 
    b_val=np.array([1.0, 0.3]), 
    f=10.0, 
    x0=np.array([0.3, 0.5]), 
    n=64, 
    use_supg=True,
):
    # Création du maillage
    msh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, n, n,
        cell_type=dolfinx.mesh.CellType.triangle
    )

    
    # Espaces de fonctions
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    
    # Fonctions test et essai
    u = TrialFunction(V)
    v = TestFunction(V)
    
    # Coefficients de l'équation
    b = ufl.as_vector(b_val)
    
    # Forme bilinéaire standard
    a_std = D * dot(grad(u), grad(v)) * dx + dot(b, grad(u)) * v * dx # type: ignore
    
    # Calcul du paramètre de stabilisation SUPG
    if use_supg:
        h = CellDiameter(msh)
        b_norm = sqrt(dot(b, b))
        tau = conditional(gt(b_norm, 1e-10), h / (2 * b_norm), 0.0)
        a_supg = tau * dot(b, grad(v)) * dot(b, grad(u)) * dx
    else:
        a_supg = 0.0
    a = a_std + a_supg

    # L(v) = 0 pour l'instant — la source ponctuelle sera injectée dans le RHS
    zero = dolfinx.fem.Constant(msh, PETSc.ScalarType(0.0))
    L = zero * v * dx

    # Dirichlet u=0 sur tout le bord
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    boundary_dofs   = dolfinx.fem.locate_dofs_topological(
        V, msh.topology.dim - 1, boundary_facets
    )
    bc = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)


    a_form = dolfinx.fem.form(a)
    L_form = dolfinx.fem.form(L)

    # Matrice de rigidité
    A = dolfinx.fem.petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    # Vecteur RHS
    b_rhs = dolfinx.fem.petsc.assemble_vector(L_form)

    # Appliquer les corrections Dirichlet sur le RHS
    dolfinx.fem.petsc.apply_lifting(b_rhs, [a_form], bcs=[[bc]])
    b_rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)


    # Injecter la source ponctuelle dans le RHS
    add_point_source(b_rhs, V, x0, f)
    # Imposer les valeurs Dirichlet dans le RHS
    dolfinx.fem.petsc.set_bc(b_rhs, [bc])

    u_sol = dolfinx.fem.Function(V)

    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("gmres")        # GMRES : adapté aux matrices non-symétriques
    ksp.getPC().setType("ilu")  # préconditionneur ILU
    ksp.setTolerances(rtol=1e-10)
    ksp.setFromOptions()

    ksp.solve(b_rhs, u_sol.x.petsc_vec)
    u_sol.x.scatter_forward()

    return u_sol

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
    print(grid.shape)
    plt.imsave("plots/convdiff_grid.png", grid, cmap="hot", vmin=0.0, vmax=grid.max())
    print("Grille sauvegardée → plots/convdiff_grid.png")


