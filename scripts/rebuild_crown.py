"""Rebuild a crown of hair that the frame cut off, then add the top margin.

Usage:  python3 scripts/rebuild_crown.py IN OUT [--px 55]

Where the hair runs into the top edge the silhouette is re-fitted from the
curve on either side of the cut, the missing sliver is filled with the hair's
own vertical tone trend plus its mirrored texture, and the outline is feathered
against the background that top_margin.py synthesises above it.
"""
import sys
import numpy as np
from PIL import Image
from gauss import gauss1d
import top_margin as tm

DARK = 60           # how much darker than background counts as hair
FEATHER = 1.4       # px of softness on the rebuilt outline


def hair_top(A, bg):
    g = A.mean(axis=2)
    dark = g < (bg - DARK)
    return np.where(dark.any(0), dark.argmax(0), A.shape[0]), dark


def crown_curve(top, span, W, reach=200):
    """Parabola through the silhouette either side of the cut."""
    lo, hi = span.min(), span.max()
    sides = np.r_[np.arange(max(0, lo - reach), lo), np.arange(hi + 1, min(W, hi + 1 + reach))]
    sides = sides[(top[sides] > 2) & (top[sides] < 300)]
    p = np.polyfit(sides, top[sides], 2)
    if p[0] <= 0:                       # must curve downwards away from the apex
        raise SystemExit("crown silhouette does not fit a head shape, do this one by hand")
    return p


def rebuild(src, dst, px=tm.DEFAULT_PX):
    A = np.asarray(Image.open(src).convert('RGB')).astype(np.float64)
    H, W, _ = A.shape
    bg = np.median(A.mean(axis=2)[:3])
    top, dark = hair_top(A, bg)
    span = np.where(top <= 1)[0]
    if not len(span):
        raise SystemExit(f"{src}: nothing is cut off at the top, use top_margin.py")

    p = crown_curve(top, span, W)
    apex = np.polyval(p, -p[1] / (2 * p[0]))
    crown = int(np.ceil(-apex)) + 4 if apex < 0 else 4
    T = int(round(px * H / float(H - px)))          # margin, oversized for the resample
    N = T + crown

    depth = tm.background_depth(A)
    depth = np.maximum(depth, 1)                    # the cut columns have no clear row
    B = tm.extend_top(A, N, depth)                  # background above the whole thing

    # columns to rebuild: wherever the fitted crown sits above the hair we can see
    x = np.arange(W)
    curve = np.polyval(p, x)
    edge = np.zeros(W, bool)
    edge[max(0, span.min() - 6):min(W, span.max() + 7)] = True   # the cut, plus a little overlap
    fill = np.where(edge & (curve < top - 0.5))[0]

    src_rows = 26                                   # hair sampled below the old edge
    for xi in fill:
        y0 = int(np.floor(curve[xi] + N))           # new silhouette
        y1 = int(top[xi]) + N                       # where real hair starts
        if y1 <= y0 or y1 - y0 > src_rows:
            continue
        col = B[y1:y1 + src_rows, xi, :]
        base = col[:6].mean(axis=0)
        trend = np.clip((col[6:12].mean(axis=0) - base) / 6.0, -1.5, 1.5)
        n = y1 - y0
        ys = np.arange(n, 0, -1)[:, None]           # distance above the real hair
        smooth = base[None, :] - trend[None, :] * ys
        detail = col[:n][::-1] - gauss1d(col[:n, :].mean(axis=1), 3)[::-1, None]
        B[y0:y1, xi, :] = np.clip(smooth + detail * 0.8, 0, 255)

    # feather the rebuilt outline so it does not read as a cut edge
    yy = np.arange(N + 40)[:, None]
    alpha = np.clip(((yy - (curve[None, :] + N)) / FEATHER) + 0.5, 0, 1)
    touched = np.zeros(W, bool)
    touched[fill] = True
    band = B[:N + 40, :, :]
    bgband = tm.extend_top(A, N, depth)[:N + 40, :, :]
    a = np.where(touched[None, :], alpha, 1.0)[:, :, None]
    B[:N + 40, :, :] = band * a + bgband * (1 - a)

    # widen to keep the aspect ratio, then resample back to the delivered size
    side = int(round(N * W / float(H)))
    c0, c1 = tm.subject_cols(A, depth)
    lm, rm = c0, W - c1
    left = int(round(side * lm / float(lm + rm))) if lm + rm else side // 2
    B = tm.extend_side(tm.extend_side(B, left, True), side - left, False)

    out = Image.fromarray(np.clip(np.rint(B), 0, 255).astype(np.uint8))
    out = out.resize((W, int(round(out.size[1] * W / out.size[0]))), Image.LANCZOS)
    out = out.crop((0, out.size[1] - H, W, out.size[1]))   # keep the frame height
    if dst.lower().endswith(('.jpg', '.jpeg')):
        out.save(dst, quality=97, subsampling=0)
    else:
        out.save(dst)

    V = np.asarray(Image.open(dst).convert('RGB')).astype(np.float64)
    nt, _ = hair_top(V, np.median(V.mean(axis=2)[:3]))
    print(f"{dst}: rebuilt {crown - 4}px of crown over {len(fill)} columns, +{T}px margin, "
          f"clearance 0 -> {int(nt.min())}px, {out.size[0]}x{out.size[1]}")


if __name__ == '__main__':
    argv = sys.argv[1:]
    px, args = tm.DEFAULT_PX, []
    i = 0
    while i < len(argv):
        if argv[i] == '--px':
            i += 1
            px = int(argv[i])
        elif argv[i].startswith('--px='):
            px = int(argv[i].split('=', 1)[1])
        else:
            args.append(argv[i])
        i += 1
    if len(args) != 2:
        raise SystemExit(__doc__)
    rebuild(args[0], args[1], px)
