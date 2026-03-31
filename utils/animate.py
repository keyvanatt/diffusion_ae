import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib
import matplotlib.colors as mcolors


def animate(
    frames: np.ndarray,
    output_path: str,
    fps: int = 10,
    cmap: str = "RdBu_r",
    label: str = "value",
    X: np.ndarray = None,
    Y: np.ndarray = None,
    title_fn=None,
) -> None:
    """Save a 3D [time, x, y] numpy array as a GIF.

    Args:
        frames:     Array of shape (T, H, W).
        output_path: Destination path for the GIF (e.g. 'output/sim.gif').
        fps:        Frames per second.
        cmap:       Matplotlib colormap name.
        label:      Colorbar label.
        X, Y:       Optional 2D meshgrid arrays for spatial axes. When provided,
                    contourf is used instead of imshow.
        title_fn:   Optional callable (t: int) -> str for per-frame titles.
    """
    vmin, vmax = frames.min(), frames.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = matplotlib.colormaps[cmap]

    use_contour = X is not None and Y is not None

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")

    if use_contour:
        ax.contourf(X, Y, frames[0], levels=50, cmap=cmap_obj, norm=norm)
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_obj), ax=ax, label=label)
    else:
        im = ax.imshow(frames[0], cmap=cmap_obj, norm=norm, origin="lower")
        plt.colorbar(im, ax=ax, label=label)

    title = ax.set_title(title_fn(0) if title_fn else "t = 0")

    def update(t):
        if use_contour:
            for c in ax.collections:
                c.remove()
            ax.contourf(X, Y, frames[t], levels=50, cmap=cmap_obj, norm=norm)
        else:
            im.set_data(frames[t])
        title.set_text(title_fn(t) if title_fn else f"t = {t}")

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 // fps, blit=False
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ani.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Saved: {output_path}")


def animate_comparaison(
    frames_a: np.ndarray,
    frames_b: np.ndarray,
    output_path: str,
    fps: int = 10,
    cmap: str = "RdBu_r",
    label: str = "value",
    X: np.ndarray = None,
    Y: np.ndarray = None,
    title_a: str = "Original",
    title_b: str = "Reconstruit",
    title_err: str = "|Erreur|",
    title_fn=None,
) -> None:
    """Save side-by-side comparison GIF: frames_a | frames_b | |frames_a - frames_b|.

    Args:
        frames_a:    Reference array of shape (T, H, W).
        frames_b:    Compared array of shape (T, H, W).
        output_path: Destination path for the GIF.
        fps:         Frames per second.
        cmap:        Colormap for the two main panels.
        label:       Colorbar label for the two main panels.
        X, Y:        Optional 2D meshgrid arrays. When provided, contourf is used.
        title_a:     Title for the first panel.
        title_b:     Title for the second panel.
        title_err:   Title for the error panel.
        title_fn:    Optional callable (t: int) -> str for the figure suptitle.
    """
    err = np.abs(frames_a - frames_b)

    vmin = min(frames_a.min(), frames_b.min())
    vmax = max(frames_a.max(), frames_b.max())
    norm_ab  = mcolors.Normalize(vmin=vmin, vmax=vmax)
    norm_err = mcolors.Normalize(vmin=0, vmax=err.max())
    cmap_ab  = matplotlib.colormaps[cmap]
    cmap_err = matplotlib.colormaps["hot_r"]

    use_contour = X is not None and Y is not None

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax_a, ax_b, ax_e = axes
    for ax in axes:
        ax.set_aspect("equal")

    if use_contour:
        ax_a.contourf(X, Y, frames_a[0], levels=50, cmap=cmap_ab, norm=norm_ab)
        ax_b.contourf(X, Y, frames_b[0], levels=50, cmap=cmap_ab, norm=norm_ab)
        ax_e.contourf(X, Y, err[0],      levels=50, cmap=cmap_err, norm=norm_err)
    else:
        im_a = ax_a.imshow(frames_a[0], cmap=cmap_ab,  norm=norm_ab,  origin="lower")
        im_b = ax_b.imshow(frames_b[0], cmap=cmap_ab,  norm=norm_ab,  origin="lower")
        im_e = ax_e.imshow(err[0],      cmap=cmap_err, norm=norm_err, origin="lower")

    plt.colorbar(cm.ScalarMappable(norm=norm_ab,  cmap=cmap_ab),  ax=ax_a, label=label)
    plt.colorbar(cm.ScalarMappable(norm=norm_ab,  cmap=cmap_ab),  ax=ax_b, label=label)
    plt.colorbar(cm.ScalarMappable(norm=norm_err, cmap=cmap_err), ax=ax_e, label="|err|")

    ax_a.set_title(title_a)
    ax_b.set_title(title_b)
    ax_e.set_title(title_err)

    suptitle = fig.suptitle(title_fn(0) if title_fn else "t = 0")

    def update(t):
        if use_contour:
            for ax in axes:
                for c in ax.collections:
                    c.remove()
            ax_a.contourf(X, Y, frames_a[t], levels=50, cmap=cmap_ab,  norm=norm_ab)
            ax_b.contourf(X, Y, frames_b[t], levels=50, cmap=cmap_ab,  norm=norm_ab)
            ax_e.contourf(X, Y, err[t],      levels=50, cmap=cmap_err, norm=norm_err)
        else:
            im_a.set_data(frames_a[t])
            im_b.set_data(frames_b[t])
            im_e.set_data(err[t])
        suptitle.set_text(title_fn(t) if title_fn else f"t = {t}")

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames_a), interval=1000 // fps, blit=False
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ani.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'Results')
    concentration = np.load(os.path.join(results_dir, "ch4_rotated.npy"))  # (cases, T, H, W)
    doe = np.load(os.path.join(results_dir, "doe_rotated.npy"))             # (cases, 4)

    case_idx = 2
    frames = concentration[case_idx]  # (T, H, W)
    doe = doe[case_idx]
    print(f"Case {case_idx}, DOE param = {doe}")
    print(f"Frames shape: {frames.shape}")

    x = np.linspace(-100, 100, frames.shape[2])
    y = np.linspace(-100, 100, frames.shape[1])
    X, Y = np.meshgrid(x, y)

    output_path = os.path.join("plots", f"CH4_case{case_idx}.gif")
    animate(
        frames,
        output_path,
        fps=10,
        cmap="RdBu_r",
        label="CH4 Concentration",
        X=X,
        Y=Y,
        title_fn=lambda t: f"Case {case_idx} | t={t} | DOE param = {doe}",
    )
