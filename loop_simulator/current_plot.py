import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# ============================================================
# LOAD FASTHENRY FILE
# ============================================================
X_length = float(input("Enter X_length (um): "))
Y_length = 5
file = f"{int(X_length)}x{int(Y_length)}.mat"

data = np.loadtxt(file)

x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

Jx = data[:, 3]
Jy = data[:, 4]
Jz = data[:, 5]

# ============================================================
# CURRENT MAGNITUDE
# ============================================================

J = np.sqrt(Jx**2 + Jy**2 + Jz**2)

# ============================================================
# USE ONLY z=0 PLANE
# ============================================================

z_unique = np.unique(z)

print("Unique z values:")
print(z_unique)

z0 = z_unique[np.argmin(np.abs(z_unique))]

mask = np.abs(z - z0) < 1e-15

x = x[mask]
y = y[mask]

Jx = Jx[mask]
Jy = Jy[mask]
Jz = Jz[mask]

J = J[mask]

print(f"Using z = {z0:.3e}")

# ============================================================
# NORMALIZE
# ============================================================

J /= np.max(J)

Jmax_vec = np.max(np.sqrt(Jx**2 + Jy**2))
Jxn = Jx / Jmax_vec
Jyn = Jy / Jmax_vec

# ============================================================
# INTERPOLATION GRID
# ============================================================

Nx = 500
Ny = 500

xi = np.linspace(x.min(), x.max(), Nx)
yi = np.linspace(y.min(), y.max(), Ny)

X, Y = np.meshgrid(xi, yi)

# ============================================================
# INTERPOLATE MAGNITUDE
# ============================================================

Jgrid = griddata(
    (x, y),
    J,
    (X, Y),
    method="cubic"
)

Jnearest = griddata(
    (x, y),
    J,
    (X, Y),
    method="nearest"
)

Jgrid = np.where(np.isnan(Jgrid), Jnearest, Jgrid)

# ============================================================
# INTERPOLATE VECTOR FIELD
# ============================================================

Jx_grid = griddata(
    (x, y),
    Jxn,
    (X, Y),
    method="cubic"
)

Jy_grid = griddata(
    (x, y),
    Jyn,
    (X, Y),
    method="cubic"
)

Jx_nearest = griddata(
    (x, y),
    Jxn,
    (X, Y),
    method="nearest"
)

Jy_nearest = griddata(
    (x, y),
    Jyn,
    (X, Y),
    method="nearest"
)

Jx_grid = np.where(np.isnan(Jx_grid), Jx_nearest, Jx_grid)
Jy_grid = np.where(np.isnan(Jy_grid), Jy_nearest, Jy_nearest)

# ============================================================
# REMOVE CENTRAL HOLE
#
# Hole:
#    -X_length/2 um < x < X_length/2 um
#    -Y_length/2 um < y < Y_length/2 um
# ============================================================

hole = (
    (X > -X_length*1e-6/2) &
    (X <  X_length*1e-6/2) &
    (Y > -Y_length*1e-6/2) &
    (Y <  Y_length*1e-6/2)
)

Jgrid[hole] = 0.0
Jx_grid[hole] = np.nan
Jy_grid[hole] = np.nan

# ============================================================
# TRANSVERSE CUT x=0 (cut along y)
# ============================================================

icut = np.argmin(np.abs(xi))

Jcut_x0 = Jgrid[:, icut].copy()

hole_cut_y = np.abs(yi) < Y_length*1e-6/2

Jcut_x0[hole_cut_y] = np.nan

Jcut_x0 /= np.nanmax(Jcut_x0)

# ============================================================
# TRANSVERSE CUT y=0 (cut along x)
# ============================================================

jcut = np.argmin(np.abs(yi))

Jcut_y0 = Jgrid[jcut, :].copy()

hole_cut_x = np.abs(xi) < X_length*1e-6/2

Jcut_y0[hole_cut_x] = np.nan

Jcut_y0 /= np.nanmax(Jcut_y0)

# ============================================================
# VECTOR FIELD DOWNSAMPLING
# ============================================================

skip = 15

Xq = X[::skip, ::skip]
Yq = Y[::skip, ::skip]

Jxq = Jx_grid[::skip, ::skip]
Jyq = Jy_grid[::skip, ::skip]

# remove tiny arrows
magq = np.sqrt(Jxq**2 + Jyq**2)

mask_q = magq > 0.02

Xq = Xq[mask_q]
Yq = Yq[mask_q]

Jxq = Jxq[mask_q]
Jyq = Jyq[mask_q]

# ============================================================
# PLOT
# ============================================================

fig = plt.figure(figsize=(20, 5))

gs = fig.add_gridspec(
    1,
    4,
    width_ratios=[1.6, 1.6, 1, 1]
)

ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
ax2 = fig.add_subplot(gs[2])
ax3 = fig.add_subplot(gs[3])

# ============================================================
# CURRENT MAGNITUDE MAP
# ============================================================

im = ax0.imshow(
    Jgrid,
    origin="lower",
    extent=[
        xi.min()*1e6,
        xi.max()*1e6,
        yi.min()*1e6,
        yi.max()*1e6
    ],
    aspect="equal"
)

ax0.axvline(
    0,
    color="w",
    linestyle="--",
    linewidth=2,
    alpha=0.8
)

ax0.plot(
    [-X_length/2, X_length/2, X_length/2, -X_length/2, -X_length/2],
    [-Y_length/2, -Y_length/2, Y_length/2, Y_length/2, -Y_length/2],
    'w--',
    lw=2
)

ax0.set_xlabel(r"$x$ ($\mu$m)", fontsize=14)
ax0.set_ylabel(r"$y$ ($\mu$m)", fontsize=14)
ax0.set_title(r"Normalized current density", fontsize=15)

cbar = plt.colorbar(im, ax=ax0)
cbar.set_label(r"$J/J_{\max}$", fontsize=13)

# ============================================================
# VECTOR FIELD
# ============================================================

ax1.imshow(
    Jgrid,
    origin="lower",
    extent=[
        xi.min()*1e6,
        xi.max()*1e6,
        yi.min()*1e6,
        yi.max()*1e6
    ],
    aspect="equal",
    alpha=0.8
)

ax1.quiver(
    Xq*1e6,
    Yq*1e6,
    Jxq,
    Jyq,
    pivot='mid',
    scale=25,
    width=0.003
)

ax1.plot(
    [-X_length/2, X_length/2, X_length/2, -X_length/2, -X_length/2],
    [-Y_length/2, -Y_length/2, Y_length/2, Y_length/2, -Y_length/2],
    'k--',
    lw=2
)

ax1.set_xlabel(r"$x$ ($\mu$m)", fontsize=14)
ax1.set_ylabel(r"$y$ ($\mu$m)", fontsize=14)
ax1.set_title(r"Current vector field", fontsize=15)

# ============================================================
# TRANSVERSE CUT x=0 (along y)
# ============================================================

ax2.plot(
    yi*1e6,
    Jcut_x0,
    lw=3
)

ax2.axvspan(
    -Y_length/2,
    Y_length/2,
    alpha=0.2
)

ax2.grid(True)

ax2.set_xlabel(r"$y$ ($\mu$m)", fontsize=14)
ax2.set_ylabel(r"$J(x=0,y)/J_{\max}$", fontsize=14)
ax2.set_title(r"Cut through $x=0$", fontsize=15)

# ============================================================
# TRANSVERSE CUT y=0 (along x)
# ============================================================

ax3.plot(
    xi*1e6,
    Jcut_y0,
    lw=3,
    color="tab:orange"
)

ax3.axvspan(
    -X_length/2,
    X_length/2,
    alpha=0.2
)

ax3.grid(True)

ax3.set_xlabel(r"$x$ ($\mu$m)", fontsize=14)
ax3.set_ylabel(r"$J(x,y=0)/J_{\max}$", fontsize=14)
ax3.set_title(r"Cut through $y=0$", fontsize=15)

# ============================================================
# STYLE
# ============================================================

for ax in [ax0, ax1, ax2, ax3]:
    ax.tick_params(labelsize=12)

plt.tight_layout()

plt.savefig(
    "current_interp_and_vectors.pdf",
    bbox_inches="tight"
)