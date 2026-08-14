from pathlib import Path
import re
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def parse_fasthenry_inp(inp_path):
    """
    Parse nodes, segments, .equiv, .external, and .Units from a FastHenry .inp file.
    """
    inp_path = Path(inp_path)
    lines = inp_path.read_text().splitlines()

    nodes = {}
    segments = []
    equiv_groups = []
    externals = []
    units = None

    node_re = re.compile(
        r'^(N\S*)\s+.*?x=([-\d.eE+]+)\s+y=([-\d.eE+]+)\s+z=([-\d.eE+]+)'
    )
    seg_re = re.compile(
        r'^(E\S*)\s+(\S+)\s+(\S+)'
        r'(?:\s+w=([-\d.eE+]+))?'
        r'(?:\s+h=([-\d.eE+]+))?'
        r'(?:\s+lambda=([-\d.eE+]+))?'
        r'(?:\s+nwinc=(\d+))?'
        r'(?:\s+nhinc=(\d+))?'
    )
    equiv_re = re.compile(r'^\.equiv\s+(.+)$')
    ext_re = re.compile(r'^\.external\s+(.+)$')
    units_re = re.compile(r'^\.Units\s+(\S+)$', re.IGNORECASE)

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("*"):
            continue

        m = units_re.match(line)
        if m:
            units = m.group(1)
            continue

        m = node_re.match(line)
        if m:
            name, x, y, z = m.groups()
            nodes[name] = (float(x), float(y), float(z))
            continue

        m = seg_re.match(line)
        if m:
            name, n1, n2, w, h, lam, nwinc, nhinc = m.groups()
            segments.append(
                {
                    "name": name,
                    "n1": n1,
                    "n2": n2,
                    "w": float(w) if w is not None else None,
                    "h": float(h) if h is not None else None,
                    "lambda": float(lam) if lam is not None else None,
                    "nwinc": int(nwinc) if nwinc is not None else None,
                    "nhinc": int(nhinc) if nhinc is not None else None,
                }
            )
            continue

        m = equiv_re.match(line)
        if m:
            equiv_groups.append(m.group(1).split())
            continue

        m = ext_re.match(line)
        if m:
            externals.extend(m.group(1).split())
            continue

    return nodes, segments, equiv_groups, externals, units


def unit_scale(units):
    """
    Scale coordinates to microns for display.
    """
    if units is None:
        return 1.0
    u = units.lower()
    if u == "um":
        return 1.0
    if u == "nm":
        return 1e-3
    if u == "mm":
        return 1e3
    if u == "cm":
        return 1e4
    if u == "m":
        return 1e6
    return 1.0


def rectangle_corners(p1, p2, width):
    """
    Return the four corners of the rectangle representing a FastHenry segment.

    p1, p2 : endpoints of the segment centerline in the XY plane
    width  : segment width (FastHenry w parameter)

    The rectangle is centered on the line from p1 to p2 and has total width = width.
    """
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    dx = x2 - x1
    dy = y2 - y1
    L = math.hypot(dx, dy)

    if L == 0:
        return None

    # Unit vectors along and perpendicular to the segment
    ux = dx / L
    uy = dy / L
    px = -uy
    py = ux

    hw = 0.5 * width

    # Four corners in order
    c1 = (x1 + px * hw, y1 + py * hw)
    c2 = (x2 + px * hw, y2 + py * hw)
    c3 = (x2 - px * hw, y2 - py * hw)
    c4 = (x1 - px * hw, y1 - py * hw)

    return [c1, c2, c3, c4]


def plot_fasthenry_rectangles(
    inp_path,
    scale_to_um=True,
    show_nodes=True,
    show_labels=False,
    edgecolor="black",
    facecolor="none",
    alpha=1.0,
    linewidth=1.2,
    figsize=(8, 8),
    title=None,
):
    """
    Plot the actual rectangles corresponding to each FastHenry segment.
    """
    nodes, segments, equiv_groups, externals, units = parse_fasthenry_inp(inp_path)
    scale = unit_scale(units) if scale_to_um else 1.0

    # Scale node coordinates
    scaled_nodes = {
        name: (x * scale, y * scale, z * scale)
        for name, (x, y, z) in nodes.items()
    }

    fig, ax = plt.subplots(figsize=figsize)

    # Draw each segment as a rectangle
    for seg in segments:
        n1, n2 = seg["n1"], seg["n2"]
        if n1 not in scaled_nodes or n2 not in scaled_nodes:
            continue

        p1 = scaled_nodes[n1]
        p2 = scaled_nodes[n2]

        w = seg["w"]
        if w is None:
            w = 0.5  # fallback if not specified

        poly = rectangle_corners(p1, p2, w)
        if poly is None:
            continue

        patch = Polygon(
            poly,
            closed=True,
            edgecolor=edgecolor,
            facecolor=facecolor,
            alpha=alpha,
            linewidth=linewidth,
        )
        ax.add_patch(patch)

        if show_labels:
            mx = 0.5 * (p1[0] + p2[0])
            my = 0.5 * (p1[1] + p2[1])
            ax.text(mx, my, seg["name"], fontsize=7, ha="center", va="center")

    # Optionally show nodes
    if show_nodes:
        xs = [p[0] for p in scaled_nodes.values()]
        ys = [p[1] for p in scaled_nodes.values()]
        ax.scatter(xs, ys, s=12, zorder=3)

    # Mark external nodes
    for n in externals:
        if n in scaled_nodes:
            x, y, _ = scaled_nodes[n]
            ax.scatter([x], [y], s=80, marker="s", zorder=4)

    # Mark equivalent groups very lightly
    for group in equiv_groups:
        coords = [scaled_nodes[n] for n in group if n in scaled_nodes]
        if not coords:
            continue
        cx = sum(c[0] for c in coords) / len(coords)
        cy = sum(c[1] for c in coords) / len(coords)
        ax.scatter([cx], [cy], s=30, zorder=4)

    # Formatting
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (um)" if scale_to_um else "x")
    ax.set_ylabel("y (um)" if scale_to_um else "y")
    if title is None:
        title = f"FastHenry rectangles: {Path(inp_path).name}"
    #ax.set_title(title)

    # Autoscale with a small margin
    all_x = [p[0] for p in scaled_nodes.values()]
    all_y = [p[1] for p in scaled_nodes.values()]
    if all_x and all_y:
        xmin, xmax = min(all_x), max(all_x)
        ymin, ymax = min(all_y), max(all_y)
        dx = xmax - xmin
        dy = ymax - ymin
        margin = 0.08 * max(dx, dy, 1.0)
        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(ymin - margin, ymax + margin)

    plt.tight_layout()
    plt.savefig(f"{Path(filename).stem}"+"_mesh.pdf")


if __name__ == "__main__":
    filename = input(
        "Enter the .inp file name [rectangular_loop_rounded.inp]: "
    ).strip()

    if not filename:
        filename = "rectangular_loop_rounded.inp"

    plot_fasthenry_rectangles(
        filename,
        scale_to_um=True,
        show_nodes=False,
        show_labels=False,
        linewidth=0.8,
    )