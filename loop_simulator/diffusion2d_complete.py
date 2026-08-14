"""
Same 2D vector diffusion equation

    d m / dt = D * Laplacian_2D ( m - chi * B )

but now solved on a RECTANGULAR LOOP geometry: a square/rectangular
frame of outer size Lx x Ly with a rectangular hole cut out of the
middle, leaving a strip ("wire") of width W all around.

Approach: build a boolean mask `inside` marking grid points that belong
to the conducting/magnetic material (the frame). The Laplacian is only
evaluated using neighbors that are also `inside`; neighbors outside the
material (in the hole or in the exterior) are excluded, which enforces
a natural zero-flux (Neumann) condition on ALL material boundaries —
both the outer edge of the loop and the inner edge of the hole.
m is simply left at 0 (or undefined/NaN for plotting) outside the mask.

This is done with a "masked Laplacian": at each material point, average
only over the material neighbors that exist, and use the count of valid
neighbors in place of a fixed factor of 4 (standard trick for irregular
domains / Neumann BC via ghost-point reflection).
"""

import numpy as np  # number of steps over which dt ramps up

def dt_factor(step, N_ramp = 1000):
    return (np.log1p(step) / np.log1p(N_ramp))**25



def build_rectangular_loop_mask(X, Y, Lx, Ly, W, x0=None, y0=None):
    """
    Build a boolean mask (Nx, Ny) that is True on a rectangular loop
    (frame) of outer dimensions Lx x Ly, wall width W, centered at
    (x0, y0) (defaults to the domain center).

    X, Y : meshgrid arrays (Nx, Ny), from np.meshgrid(x, y, indexing="ij")
    Lx, Ly : outer dimensions of the loop
    W : width of the frame walls (same on all 4 sides)
    """
    if x0 is None:
        x0 = 0.5 * (X.min() + X.max())
    if y0 is None:
        y0 = 0.5 * (Y.min() + Y.max())

    # Outer rectangle: |x-x0| <= Lx/2 and |y-y0| <= Ly/2
    outer = (np.abs(X - x0) <= Lx / 2) & (np.abs(Y - y0) <= Ly / 2)

    # Inner rectangle (the hole): shrink by wall width W on each side
    inner = (np.abs(X - x0) <= Lx / 2 - W) & (np.abs(Y - y0) <= Ly / 2 - W)

    mask = outer & (~inner)
    return mask


def masked_laplacian2d(f, mask, dx, dy):
    """
    Discrete Laplacian of f (Nx, Ny) restricted to `mask` (True = material).
    Uses a variable-stencil finite-volume-style formula so that boundaries
    of the mask (outer edge AND inner hole edge) behave like zero-flux
    (Neumann) boundaries automatically: neighbors outside the mask are
    simply excluded from the average (equivalent to a mirror/ghost point
    equal to the center value, i.e. zero gradient across the boundary).

    Returns an array the same shape as f; values outside the mask are 0.
    """
    Nx, Ny = f.shape
    lap = np.zeros_like(f)

    # Shifted masks and shifted field values, with out-of-bounds = not valid
    def shifted(arr, dx_shift, dy_shift):
        """Shift array by (dx_shift, dy_shift) with 'invalid' padding."""
        out = np.zeros_like(arr)
        valid = np.zeros(arr.shape, dtype=bool)

        src_x0 = max(0, -dx_shift)
        src_x1 = min(Nx, Nx - dx_shift)
        src_y0 = max(0, -dy_shift)
        src_y1 = min(Ny, Ny - dy_shift)

        dst_x0 = max(0, dx_shift)
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y0 = max(0, dy_shift)
        dst_y1 = dst_y0 + (src_y1 - src_y0)

        out[dst_x0:dst_x1, dst_y0:dst_y1] = arr[src_x0:src_x1, src_y0:src_y1]
        valid[dst_x0:dst_x1, dst_y0:dst_y1] = True
        return out, valid

    f_xp, v_xp = shifted(f, 1, 0)
    f_xm, v_xm = shifted(f, -1, 0)
    f_yp, v_yp = shifted(f, 0, 1)
    f_ym, v_ym = shifted(f, 0, -1)

    mask_xp, _ = shifted(mask, 1, 0)
    mask_xm, _ = shifted(mask, -1, 0)
    mask_yp, _ = shifted(mask, 0, 1)
    mask_ym, _ = shifted(mask, 0, -1)

    valid_xp = v_xp & mask_xp
    valid_xm = v_xm & mask_xm
    valid_yp = v_yp & mask_yp
    valid_ym = v_ym & mask_ym

    # x-direction second derivative (Neumann: missing neighbor -> use f itself)
    fxp = np.where(valid_xp, f_xp, f)
    fxm = np.where(valid_xm, f_xm, f)
    fyp = np.where(valid_yp, f_yp, f)
    fym = np.where(valid_ym, f_ym, f)

    lap = (fxp - 2.0 * f + fxm) / dx**2 + (fyp - 2.0 * f + fym) / dy**2
    lap[~mask] = 0.0
    return lap


def step_ftcs_masked(m, B, mask, D, dx, dy, dt, T=20e-3):
    """
    Advance m (Nx, Ny, 3) by one explicit Euler step, restricted to `mask`.
    Points outside the mask are held at 0.
    """
    m_new = np.zeros_like(m)
    meq = np.zeros_like(m)
    Bx=B[...,0]
    By=B[...,1]
    Bz=B[...,2]
    B2 = Bx**2 + By**2 + Bz**2
    eps=1e-30
    B2_safe = np.where(B2 > eps, B2, .001)
    Bmag=np.sqrt(B2_safe)
    mu_B = 9.2740100783e-24   # J/T (equivalently A·m²)
    k_B  = 1.380649e-23       # J/K
    g=1
    x = g * mu_B * Bmag / (2 * k_B * T)
    #meq=np.tanh(x)*B/Bmag
    meq[...,0]=np.tanh(x)*Bx/Bmag
    meq[...,1]=np.tanh(x)*By/Bmag
    meq[...,2]=np.tanh(x)*Bz/Bmag

    for i in range(3):
        lap_m = masked_laplacian2d(m[..., i]-meq[..., i], mask, dx, dy)
        lap_B = masked_laplacian2d(B[..., i], mask, dx, dy)
        m_new[..., i] = np.where(
            mask, m[..., i] + dt * D * (lap_m), 0.0
        )
    return m_new


# def max_stable_dt(D, dx, dy, safety=0.9):
#     return safety * 0.5 / (D * (1.0 / dx**2 + 1.0 / dy**2))


def max_stable_dt(D, dx, dy, Bmax=0, gamma=1.76e11,
           T1=np.inf, T2=np.inf, safety=0.9):

    # Diffusion limit
    dt_diff = 0.5 / (D * (1.0/dx**2 + 1.0/dy**2))

    # # Precession limit
    # if gamma == 0 or Bmax <= 0:
    #     dt_prec = np.inf
    # else:
    #     dt_prec = 0.1 / (gamma * Bmax)

    # Relaxation limits
    dt_T1 = np.inf if np.isinf(T1) else T1
    dt_T2 = np.inf if np.isinf(T2) else T2

    return safety * min(dt_diff, dt_T1, dt_T2)


import numpy as np
def build_nonuniform_time_grid(t_final, T1, n_points, n_frac_before_T1=0.6, t_min=1e-15):
    """
    Dense near t=0 (log-spaced), coarser for t > T1 (linear or log).
    t_min: smallest nonzero time scale (avoid log(0)).
    """
    n1 = int(n_points * n_frac_before_T1)
    n2 = n_points - n1

    # Log-spaced from t_min to T1: naturally dense near small t
    t_part1 = np.geomspace(t_min, T1, n1)
    t_part1 = np.concatenate([[0.0], t_part1])  # include exact t=0

    # Linear (or log again) from T1 to t_final
    t_part2 = np.linspace(T1, t_final, n2)

    t_grid = np.concatenate([t_part1, t_part2])
    return np.unique(t_grid)
def solve_diffusion_2d_loop(
    m0, B_func, mask, D, dx, dy, t_grid, 
    T=20e-3,
    T1=0.2, T2=np.inf, gamma=1.76e11,
    diffusion=True, precession=False, relaxation=True,
    dt_max=None, safety=0.9, min_substeps=1
):
    """
    Time evolution of magnetization on a masked (loop) geometry, saving
    output at the (possibly non-uniform) times specified in t_grid.

    Internally, the integrator always advances with a stable internal
    dt (<= dt_max), taking multiple substeps between consecutive
    entries of t_grid whenever the requested output interval would
    otherwise violate the FTCS stability condition. This makes the
    integration robust regardless of how coarse/non-uniform t_grid is.

    Parameters
    ----------
    t_grid : array, monotonically increasing, t_grid[0] = 0.
             Output (saved) times. Internal integration dt is
             independent of the spacing of t_grid.
    dt_max : float, optional
             Maximum stable internal dt. If None, computed automatically
             from max_stable_dt(D, dx, dy, T1=T1, T2=T2, safety=safety).
    min_substeps : int
             Minimum number of substeps to take per output interval,
             even if a single stable dt would already cover it (useful
             if you want a floor on temporal resolution regardless of
             CFL).
    """
    if dt_max is None:
        dt_max = max_stable_dt(D, dx, dy, T1=T1, T2=T2, safety=safety)

    m = m0.copy()
    m[~mask] = 0.0

    times = [t_grid[0]]
    m_hist = [m.copy()]

    invT1 = 0.0 if np.isinf(T1) else 1.0 / T1
    invT2 = 0.0 if np.isinf(T2) else 1.0 / T2
    eps = 1e-35
    mu_B = 9.2740100783e-24
    k_B = 1.380649e-23
    g = 1.0

    b_axis = np.zeros_like(m0)
    b_axis[..., 2] = 1.0

    t = t_grid[0]

    for n in range(1, len(t_grid)):
        t_target = t_grid[n]
        interval = t_target - t

        if interval <= 0:
            # Degenerate/duplicate grid point; just re-save current state
            times.append(t_target)
            m_hist.append(m.copy())
            continue

        # Number of stable substeps needed to cover this output interval
        n_sub = max(int(np.ceil(interval / dt_max)), min_substeps)
        dt_internal = interval / n_sub  # exact division -> lands exactly on t_target

        for _ in range(n_sub):
            B = B_func(t)

            # ---------------- Diffusion ----------------
            if diffusion:
                m = step_ftcs_masked(m, B, mask, D, dx, dy, dt_internal, T)

            # ---------------- Bloch relaxation (field-following axis) ----------------
            if relaxation:
                mx, my, mz = m[..., 0], m[..., 1], m[..., 2]
                Bx, By, Bz = B[..., 0], B[..., 1], B[..., 2]

                B2 = Bx**2 + By**2 + Bz**2
                zeroB = (B2 <= eps)
                Bmag = np.sqrt(B2)
                Bmag_safe = np.where(zeroB, 1.0, Bmag)

                bx_inst = Bx / Bmag_safe
                by_inst = By / Bmag_safe
                bz_inst = Bz / Bmag_safe

                b_axis[..., 0] = np.where(zeroB, b_axis[..., 0], bx_inst)
                b_axis[..., 1] = np.where(zeroB, b_axis[..., 1], by_inst)
                b_axis[..., 2] = np.where(zeroB, b_axis[..., 2], bz_inst)

                bx, by, bz = b_axis[..., 0], b_axis[..., 1], b_axis[..., 2]

                x = g * mu_B * Bmag / (2 * k_B * T)
                tanhx = np.tanh(x)

                mxeq, myeq, mzeq = tanhx * bx, tanhx * by, tanhx * bz
                mdotB = mx * bx + my * by + mz * bz
                projx, projy, projz = mdotB * bx, mdotB * by, mdotB * bz

                m[..., 0] += dt_internal * (-mx * invT2 + mxeq * invT1 + projx * (invT2 - invT1))
                m[..., 1] += dt_internal * (-my * invT2 + myeq * invT1 + projy * (invT2 - invT1))
                m[..., 2] += dt_internal * (-mz * invT2 + mzeq * invT1 + projz * (invT2 - invT1))

            # ---------------- Larmor precession ----------------
            if precession:
                mx, my, mz = m[..., 0], m[..., 1], m[..., 2]
                Bx, By, Bz = B[..., 0], B[..., 1], B[..., 2]

                cross_x = my * Bz - mz * By
                cross_y = mz * Bx - mx * Bz
                cross_z = mx * By - my * Bx

                m[..., 0] += gamma * dt_internal * cross_x
                m[..., 1] += gamma * dt_internal * cross_y
                m[..., 2] += gamma * dt_internal * cross_z

            m[~mask] = 0.0
            t += dt_internal

        # Snap exactly to the target grid time (avoid float drift)
        t = t_target
        times.append(t)
        m_hist.append(m.copy())

    return np.array(times), np.array(m_hist)

# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Grid covering a bit more than the outer loop size
    Nx, Ny = 120, 120
    domain_size = 1.2
    x = np.linspace(-domain_size / 2, domain_size / 2, Nx)
    y = np.linspace(-domain_size / 2, domain_size / 2, Ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Loop geometry
    Lx, Ly = 0.8, 0.6     # outer dimensions
    W = 0.1                # wall width

    mask = build_rectangular_loop_mask(X, Y, Lx, Ly, W)

    # Physical parameters
    D = 1.0
    chi = 0.5

    # Initial condition: mz = 1 on one arm of the loop, 0 elsewhere
    m0 = np.zeros((Nx, Ny, 3))
    hot_spot = mask & (X < -Lx / 2 + W) & (np.abs(Y) < Ly / 4)
    m0[hot_spot, 2] = 1.0

    # External field: uniform Bz applied everywhere (e.g. external flux)
    B_static = np.zeros((Nx, Ny, 3))
    B_static[..., 2] = 0.3

    def B_func(t):
        return B_static

    dt = max_stable_dt(D, dx, dy, safety=0.9)
    n_steps = 3000
    save_every = 150

    times, m_hist = solve_diffusion_2d_loop(
        m0, B_func, mask, D, chi, dx, dy, dt, n_steps, save_every=save_every
    )

    print(f"dt = {dt:.3e}, total simulated time = {times[-1]:.3e}")
    print(f"Saved {len(times)} snapshots, m_hist shape = {m_hist.shape}")
