import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pathlib import Path
import re

# ============================================================
# SETTINGS
# ============================================================

# Interpolation grid for the z≈0 plane
Nx = 500
Ny = 500

# Cut ranges in microns
xcut_y_min_um = -10.0
xcut_y_max_um = -5.0

# For the y=0 cut, x runs from -(N+5) to -N (in microns)
# because that is outside the hole and on the left side.
npts_cut = 1500

# Output folder
outdir = Path("current")
outdir.mkdir(exist_ok=True)

# ============================================================
# FILES
# ============================================================

files = sorted(Path(".").glob("*x5.mat"))

if len(files) == 0:
    raise RuntimeError("No files matching *x5.mat found.")

def extract_N(fname):
    m = re.search(r"(\d+)x5\.mat", fname.name)
    if m is None:
        raise ValueError(f"Could not parse N from filename: {fname.name}")
    return int(m.group(1))

files = sorted(files, key=extract_N)

# ============================================================
# HELPERS
# ============================================================

def safe_griddata(points, values, XI, YI):
    """
    Interpolate on a regular grid.
    Try cubic, then linear, then nearest to fill remaining holes.
    """
    Z = None
    for method in ("cubic", "linear"):
        try:
            Z = griddata(points, values, (XI, YI), method=method)
            if Z is not None and np.any(np.isfinite(Z)):
                break
        except Exception:
            Z = None

    Z_near = griddata(points, values, (XI, YI), method="nearest")
    if Z is None:
        return Z_near
    return np.where(np.isnan(Z), Z_near, Z)

def load_z0_plane(file):
    """
    Load the file and keep only the plane closest to z=0.
    """
    data = np.loadtxt(file)

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    Jx = data[:, 3]
    Jy = data[:, 4]
    Jz = data[:, 5]

    z_unique = np.unique(z)
    z0 = z_unique[np.argmin(np.abs(z_unique))]
    mask = np.abs(z - z0) < 1e-15

    xx = x[mask]
    yy = y[mask]
    Jx0 = Jx[mask]
    Jy0 = Jy[mask]
    Jz0 = Jz[mask]

    if len(xx) < 10:
        raise RuntimeError(f"Not enough points in z≈0 plane for {file.name}")

    return xx, yy, Jx0, Jy0, Jz0, z0

def interpolate_current_plane(xx, yy, Jx0, Jy0, Jz0, N):
    """
    Interpolate Jx, Jy, Jz and J on a 2D grid, then mask the hole.
    Returns the grid and the grid coordinates.
    """
    points = np.column_stack([xx, yy])

    xi = np.linspace(xx.min(), xx.max(), Nx)
    yi = np.linspace(yy.min(), yy.max(), Ny)
    X, Y = np.meshgrid(xi, yi)

    # Interpolate vector components
    Jx_grid = safe_griddata(points, Jx0, X, Y)
    Jy_grid = safe_griddata(points, Jy0, X, Y)
    Jz_grid = safe_griddata(points, Jz0, X, Y)

    # Current magnitude from interpolated components
    J_grid = np.sqrt(Jx_grid**2 + Jy_grid**2 + Jz_grid**2)

    # Hole mask
    hole = (
        (X > -N/2 * 1e-6) &
        (X <  N/2 * 1e-6) &
        (Y > -2.5e-6) &
        (Y <  2.5e-6)
    )

    # Keep the hole empty in the plots and in the saved data
    Jx_grid = np.where(hole, 0, Jx_grid)
    Jy_grid = np.where(hole, 0, Jy_grid)
    Jz_grid = np.where(hole, 0, Jz_grid)
    J_grid = np.where(hole, 0, J_grid)

    return X, Y, xi, yi, Jx_grid, Jy_grid, Jz_grid, J_grid

def save_interpolation(file, X, Y, Jx_grid, Jy_grid, Jz_grid, J_grid):
    """
    Save interpolated data as plain text with .mat extension.
    Columns: x y Jx Jy Jz J
    """
    out = np.column_stack([
        X.ravel(),        # x [m]
        Y.ravel(),        # y [m]
        Jx_grid.ravel(),  # Jx [A/m^2]
        Jy_grid.ravel(),  # Jy [A/m^2]
        Jz_grid.ravel(),  # Jz [A/m^2]
        J_grid.ravel(),   # |J| [A/m^2]
    ])

    np.savetxt(
        file.with_name(f"{file.stem}_interp.mat"),
        out,
        header="x_m y_m Jx_A_per_m2 Jy_A_per_m2 Jz_A_per_m2 J_A_per_m2"
    )

def make_cut_plots(normalize=True, filename="all_meshes_z0_cuts.pdf"):
    """
    Make one figure with one row per file and two columns:
    left = x=0 cut over y in [-10, -5] um
    right = y=0 cut over x in [-(N+5), -N] um

    The cuts are taken from the interpolated 2D grid.
    """
    nfiles = len(files)

    fig, axes = plt.subplots(
        nfiles,
        2,
        figsize=(14, 3.8 * nfiles),
        squeeze=False
    )

    for ifile, file in enumerate(files):
        print(f"Processing {file.name}")

        N = extract_N(file)

        xx, yy, Jx0, Jy0, Jz0, z0 = load_z0_plane(file)

        X, Y, xi, yi, Jx_grid, Jy_grid, Jz_grid, J_grid = interpolate_current_plane(
            xx, yy, Jx0, Jy0, Jz0, N
        )

        # Save interpolated grid data
        save_interpolation(file, X, Y, Jx_grid, Jy_grid, Jz_grid, J_grid)

        # Normalize or not
        if normalize:
            Jmax = np.nanmax(J_grid)
            if not np.isfinite(Jmax) or Jmax <= 0:
                raise RuntimeError(f"Invalid Jmax for {file.name}")
            Jplot = J_grid / Jmax
            ylabel = r"$J/J_{\max}$"
            title_suffix = "normalized"
        else:
            Jplot = J_grid.copy()
            ylabel = r"$J$ (A/m$^2$)"
            title_suffix = "raw"

        ax_x = axes[ifile, 0]
        ax_y = axes[ifile, 1]

        # ====================================================
        # CUT AT x = 0  -> plot versus y in [-10, -5] um
        # ====================================================

        icut = np.argmin(np.abs(xi))
        mask_y = (
            (yi >= xcut_y_min_um * 1e-6) &
            (yi <= xcut_y_max_um * 1e-6)
        )

        y_cut = yi[mask_y] * 1e6
        Jcut_x = Jplot[mask_y, icut]

        ax_x.plot(y_cut, Jcut_x, lw=2)
        ax_x.grid(True)
        ax_x.set_xlim(xcut_y_min_um, xcut_y_max_um)
        ax_x.set_xlabel(r"$y$ ($\mu$m)")
        ax_x.set_ylabel(ylabel)
        ax_x.set_title(rf"N={N} : cut at $x=0$ ({title_suffix}), $z \approx 0$")

        # ====================================================
        # CUT AT y = 0  -> plot versus x in [-(N+5), -N] um
        # ====================================================

        jcut = np.argmin(np.abs(yi))
        mask_x = (
            (xi >= -(N/2 + 5) * 1e-6) &
            (xi <= -N/2 * 1e-6)
        )

        x_cut = xi[mask_x] * 1e6
        Jcut_y = Jplot[jcut, mask_x]

        ax_y.plot(x_cut, Jcut_y, lw=2)
        ax_y.grid(True)
        ax_y.set_xlim(-(N/2 + 5), -N/2)
        ax_y.set_xlabel(r"$x$ ($\mu$m)")
        ax_y.set_ylabel(ylabel)
        ax_y.set_title(rf"N={N} : cut at $y=0$ ({title_suffix}), $z \approx 0$")

        if normalize:
            ax_x.set_ylim(0, 1.05)
            ax_y.set_ylim(0, 1.05)

    plt.tight_layout()
    #plt.savefig(outdir / filename, dpi=300, bbox_inches="tight")
    #plt.show()
    plt.close(fig)

# ============================================================
# MAKE BOTH FIGURES
# ============================================================

make_cut_plots(normalize=True, filename="all_meshes_z0_cuts_normalized.pdf")
#make_cut_plots(normalize=False, filename="all_meshes_z0_cuts_raw.pdf")