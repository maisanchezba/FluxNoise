from pathlib import Path
import math


def cosine_boundaries(total_width, n_slices):
    """
    n_slices slices -> n_slices+1 boundaries.
    Cosine clustering makes slices smaller near both outer edges.
    """
    if n_slices < 1:
        raise ValueError("n_slices must be >= 1")
    return [
        0.5 * total_width * (1.0 - math.cos(math.pi * j / n_slices))
        for j in range(n_slices + 1)
    ]

"""
def cosine_boundaries(total_width, n_slices, power=2.0):
    if n_slices < 1:
        raise ValueError("n_slices must be >= 1")

    return [
        0.5 * total_width * (
            1.0 - math.cos(
                math.pi * (
                    0.5 * (2 * (j / n_slices)) ** power
                    if j / n_slices <= 0.5
                    else 1.0 - 0.5 * (2 * (1.0 - j / n_slices)) ** power
                )
            )
        )
        for j in range(n_slices + 1)
    ]
"""

def graded_positions(L, ds_corner=0.03, ds_mid=0.25, corner_span=0.15):
    """
    Positions from 0 to L with fine spacing near both ends and coarser spacing in the middle.
    """
    if L <= 0:
        return [0.0]

    if L <= 2 * corner_span:
        n = max(2, math.ceil(L / ds_corner))
        return [L * i / n for i in range(n + 1)]

    pts = [0.0]

    # Left corner region
    x = 0.0
    while x + ds_corner < corner_span:
        x += ds_corner
        pts.append(x)
    if pts[-1] < corner_span:
        pts.append(corner_span)

    # Middle region
    x = pts[-1]
    mid_end = L - corner_span
    while x + ds_mid < mid_end:
        x += ds_mid
        pts.append(x)
    if pts[-1] < mid_end:
        pts.append(mid_end)

    # Right corner region
    x = pts[-1]
    while x + ds_corner < L:
        x += ds_corner
        pts.append(x)
    if pts[-1] < L:
        pts.append(L)

    # Remove duplicates from rounding
    out = [pts[0]]
    for p in pts[1:]:
        if abs(p - out[-1]) > 1e-12:
            out.append(p)
    return out


def line_points(p0, p1, svals):
    """
    Linear interpolation along a segment from p0 to p1.
    """
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    return [
        (x0 + s * (x1 - x0), y0 + s * (y1 - y0), z0 + s * (z1 - z0))
        for s in svals
    ]


def arc_points(center, radius, theta0, theta1, npts):
    """
    Points along a circular arc in the XY plane.
    """
    if npts < 2:
        raise ValueError("npts must be >= 2")

    cx, cy, cz = center
    thetas = [theta0 + (theta1 - theta0) * i / (npts - 1) for i in range(npts)]
    return [
        (cx + radius * math.cos(th), cy + radius * math.sin(th), cz)
        for th in thetas
    ]


def add_points_and_segments(
    points,
    seg_w,
    node_lines,
    seg_lines,
    cache,
    node_counter,
    seg_counter,
    h,
    lam,
    nwinc,
    nhinc,
):
    """
    Add a polyline to the FastHenry node/segment lists.
    Returns updated node_counter and seg_counter.
    """
    def new_node(coord):
        nonlocal node_counter
        key = tuple(round(v, 12) for v in coord)
        if key not in cache:
            name = f"N{node_counter}"
            node_counter += 1
            cache[key] = name
            x, y, z = coord
            node_lines.append(f"{name} x={x:g} y={y:g} z={z:g}")
        return cache[key]

    node_list = [new_node(pt) for pt in points]
    for a, b in zip(node_list[:-1], node_list[1:]):
        seg_lines.append(
            f"E{seg_counter} {a} {b} w={seg_w:g} h={h:g} lambda={lam:g} nwinc={nwinc} nhinc={nhinc}"
        )
        seg_counter += 1

    return node_counter, seg_counter, node_list


def write_rectangular_loop_inp(
    filename="rectangular_loop_rounded.inp",
    outer_half_x=7.0,
    outer_half_y=5.0,
    band_width=1.0,
    n_loops=12,
    slit_gap=0.25,
    h=0.1,
    lam=0.07,
    nwinc=1,
    nhinc=3,
    ds_corner=0.03,
    ds_mid_x=0.25,
    ds_mid_y=0.25,
    corner_span=0.15,
    corner_radius=0.05,
    n_corner_pts=5,
):
    if n_loops < 1:
        raise ValueError("n_loops must be >= 1")
    if band_width <= 0:
        raise ValueError("band_width must be positive")
    if outer_half_x <= band_width or outer_half_y <= band_width:
        raise ValueError("outer_half_x and outer_half_y must be larger than band_width")
    if corner_radius <= 0:
        raise ValueError("corner_radius must be positive")
    if n_corner_pts < 2:
        raise ValueError("n_corner_pts must be >= 2")

    boundaries = cosine_boundaries(band_width, n_loops)
    centers = [0.5 * (boundaries[i] + boundaries[i + 1]) for i in range(n_loops)]
    widths = [boundaries[i + 1] - boundaries[i] for i in range(n_loops)]

    node_lines = []
    seg_lines = []
    top_gap_nodes = []
    bot_gap_nodes = []

    node_counter = 1
    seg_counter = 1

    for i in range(n_loops):
        t = centers[i]
        w_i = widths[i]

        sx = outer_half_x - t
        sy = outer_half_y - t

        if sx <= slit_gap / 2:
            raise ValueError(
                f"Loop {i} too narrow in x: half-width {sx:g} <= slit_gap/2 = {slit_gap/2:g}"
            )
        if sy <= slit_gap / 2:
            raise ValueError(
                f"Loop {i} too short in y: half-height {sy:g} <= slit_gap/2 = {slit_gap/2:g}"
            )

        r = min(
            corner_radius,
            0.20 * min(sx, sy),
            0.45 * (sy - slit_gap / 2.0),
            0.45 * sx,
        )
        if r <= 0:
            raise ValueError(f"Loop {i} has invalid corner radius after clipping.")

        if sx - r <= 0 or sy - r <= slit_gap / 2:
            raise ValueError(
                f"Loop {i} too small for the chosen corner_radius={corner_radius:g}."
            )

        cache = {}

        # Straight lengths
        bottom_len = 2.0 * (sx - r)         # along x
        right_len = 2.0 * (sy - r)           # along y
        left_len = (sy - r) - slit_gap / 2.0 # along y on the left side

        if left_len <= 0:
            raise ValueError(
                f"Loop {i} too small for slit_gap={slit_gap:g} and corner_radius={corner_radius:g}."
            )

        # Use different mid spacings in x and y
        u_bottom = [p / bottom_len for p in graded_positions(
            bottom_len,
            ds_corner=ds_corner,
            ds_mid=ds_mid_x,
            corner_span=corner_span,
        )]

        u_side = [p / right_len for p in graded_positions(
            right_len,
            ds_corner=ds_corner,
            ds_mid=ds_mid_y,
            corner_span=corner_span,
        )]

        u_left = [p / left_len for p in graded_positions(
            left_len,
            ds_corner=ds_corner,
            ds_mid=ds_mid_y,
            corner_span=min(corner_span, left_len / 2.0),
        )]

        # Bottom straight
        bottom_pts = line_points(
            (-sx + r, -sy, 0),
            ( sx - r, -sy, 0),
            u_bottom
        )

        # Bottom-right corner
        br_arc = arc_points(
            center=(sx - r, -sy + r, 0),
            radius=r,
            theta0=-math.pi / 2,
            theta1=0.0,
            npts=n_corner_pts
        )

        # Right straight
        right_pts = line_points(
            (sx, -sy + r, 0),
            (sx,  sy - r, 0),
            u_side
        )

        # Top-right corner
        tr_arc = arc_points(
            center=(sx - r, sy - r, 0),
            radius=r,
            theta0=math.pi / 2,
            theta1=0.0,
            npts=n_corner_pts
        )

        # Top straight
        top_pts = line_points(
            (sx - r,  sy, 0),
            (-sx + r, sy, 0),
            u_bottom
        )

        # Top-left corner
        tl_arc = arc_points(
            center=(-sx + r, sy - r, 0),
            radius=r,
            theta0=math.pi / 2,
            theta1=math.pi,
            npts=n_corner_pts
        )

        # Left upper straight
        left_upper_pts = line_points(
            (-sx, sy - r, 0),
            (-sx, slit_gap / 2.0, 0),
            u_left
        )

        # Left lower straight
        left_lower_pts = line_points(
            (-sx, -slit_gap / 2.0, 0),
            (-sx, -sy + r, 0),
            u_left
        )

        # Bottom-left corner
        bl_arc = arc_points(
            center=(-sx + r, -sy + r, 0),
            radius=r,
            theta0=math.pi,
            theta1=3.0 * math.pi / 2.0,
            npts=n_corner_pts
        )

        # Add all polylines
        node_counter, seg_counter, bottom_nodes = add_points_and_segments(
            bottom_pts, w_i, node_lines, seg_lines, cache, node_counter, seg_counter,
            h, lam, nwinc, nhinc
        )
        node_counter, seg_counter, br_nodes = add_points_and_segments(
            br_arc, w_i, node_lines, seg_lines, cache, node_counter, seg_counter,
            h, lam, nwinc, nhinc
        )
        node_counter, seg_counter, right_nodes = add_points_and_segments(
            right_pts, w_i, node_lines, seg_lines, cache, node_counter, seg_counter,
            h, lam, nwinc, nhinc
        )
        node_counter, seg_counter, tr_nodes = add_points_and_segments(
            tr_arc, w_i, node_lines, seg_lines, cache, node_counter, seg_counter,
            h, lam, nwinc, nhinc
        )
        node_counter, seg_counter, top_nodes = add_points_and_segments(
            top_pts, w_i, node_lines, seg_lines, cache, node_counter, seg_counter,
            h, lam, nwinc, nhinc
        )
        node_counter, seg_counter, tl_nodes = add_points_and_segments(
            tl_arc, w_i, node_lines, seg_lines, cache, node_counter, seg_counter,
            h, lam, nwinc, nhinc
        )
        node_counter, seg_counter, leftu_nodes = add_points_and_segments(
            left_upper_pts, w_i, node_lines, seg_lines, cache, node_counter, seg_counter,
            h, lam, nwinc, nhinc
        )
        node_counter, seg_counter, leftl_nodes = add_points_and_segments(
            left_lower_pts, w_i, node_lines, seg_lines, cache, node_counter, seg_counter,
            h, lam, nwinc, nhinc
        )
        node_counter, seg_counter, bl_nodes = add_points_and_segments(
            bl_arc, w_i, node_lines, seg_lines, cache, node_counter, seg_counter,
            h, lam, nwinc, nhinc
        )

        top_gap_nodes.append(leftu_nodes[-1])
        bot_gap_nodes.append(leftl_nodes[0])

    out = []
    out.append("* Concentric rectangular superconducting loop with circular corner smoothing")
    out.append(".Units um")
    out.append(".freq fmin=1e-3 fmax=1e-3 ndec=1")
    out.append("")
    out.append("* Nodes")
    out.extend(node_lines)
    out.append("")
    out.append("* Segments")
    out.extend(seg_lines)
    out.append("")
    out.append("* Tie together all upper slit nodes")
    out.append(".equiv " + " ".join(top_gap_nodes))
    out.append("* Tie together all lower slit nodes")
    out.append(".equiv " + " ".join(bot_gap_nodes))
    out.append("")
    out.append(f".external {top_gap_nodes[0]} {bot_gap_nodes[0]}")
    out.append(".end")

    Path(filename).write_text("\n".join(out))
    return filename

"""
if __name__ == "__main__":
    fname = write_rectangular_loop_inp(
        filename="rectangular_loop_rounded.inp",
        outer_half_x=3.0,
        outer_half_y=2.0,
        band_width=1.0,
        n_loops=30,
        slit_gap=0.25,
        h=0.1,
        lam=0.07,
        nwinc=1,
        nhinc=3,
        ds_corner=0.03,
        ds_mid_x=1.6,
        ds_mid_y=0.8,
        corner_span=0.20,
        corner_radius=0.08,
        n_corner_pts=8
    )
"""

if __name__ == "__main__":
    Lx = float(input(
        "Enter the length in x for loop [5]: "
    ).strip())

    if not Lx:
        Lx = 5.
    W=5.
    Ly=5.

    fname = write_rectangular_loop_inp(
        filename=f"rectangular_loop_rounded_{int(Lx)}x{int(Ly)}.inp",
        outer_half_x=Lx/2+W,
        outer_half_y=Ly/2+W,
        band_width=W,
        n_loops=78,
        slit_gap=0.15,
        h=0.1,
        lam=0.07,
        nwinc=1,
        nhinc=3,
        ds_corner=0.12,
        ds_mid_x=.2,
        ds_mid_y=1.,
        corner_span=0.40,
        corner_radius=0.2,
        n_corner_pts=6
    )
    
    print("Wrote:", fname)