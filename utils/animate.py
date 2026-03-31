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


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    results_dir = os.path.join(os.path.dirname(__file__), "..", "Results")
    concentration = np.load(os.path.join(results_dir, "CH4.npy"))  # (cases, T, H, W)
    doe = np.load(os.path.join(results_dir, "doe.npy"))             # (cases, 3)

    case_idx = 2
    frames = concentration[case_idx]  # (T, H, W)
    D, angle, inj = doe[case_idx]
    print(f"Case {case_idx}: D={D}, angle={angle}°, injection={inj} kg/m³/s")
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
        title_fn=lambda t: f"Case {case_idx} | t={t} | D={D}, angle={angle}°",
    )
