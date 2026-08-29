"""Add a small strip of background above the subject.

Usage:  python3 scripts/top_margin.py IN OUT [--px 55] [--no-fit]

The strip is synthesised from the studio background that is already in the
frame: a per-column anchor colour taken from the real top rows, a vertical
gradient measured down the columns that stay clear of the subject, and grain
matched to the background's own noise.  The sides are widened by the same
proportion so the frame keeps its aspect ratio and the result is resampled
back to the input size, so --px is the margin you get in the delivered file.
Pass --no-fit to keep the enlarged canvas instead.
"""
import sys
import numpy as np
from PIL import Image
from gauss import gauss1d

DEFAULT_PX = 55
SLOPE_LIMIT = 0.15          # max colour drift per row, keeps extrapolation sane
GRAD_DEPTH = 120            # rows used to measure the vertical gradient


def background_depth(A):
    """Per column: how many rows from the top are still clean background."""
    g = A.mean(axis=2)
    bg = np.median(g[:3])
    is_bg = g > (bg - 18)
    blocked = ~is_bg
    depth = np.where(blocked.any(axis=0), blocked.argmax(axis=0), A.shape[0])
    return depth.astype(int)


def fill_invalid(vals, valid):
    """Interpolate a per-column series across the columns we could not measure."""
    x = np.arange(len(vals), dtype=np.float64)
    if not valid.any():
        raise SystemExit("no clean background column to sample")
    return np.interp(x, x[valid], vals[valid])


def column_slopes(A, depth):
    """Vertical colour gradient per column, measured where the view is clear."""
    H, W, C = A.shape
    n = min(GRAD_DEPTH, H // 4)
    valid = depth >= n
    if valid.sum() < W // 20:                      # subject reaches high on both sides
        n = max(6, int(depth.max() * 0.8))
        valid = depth >= n
    y = np.arange(n, dtype=np.float64)
    ym = y.mean()
    denom = ((y - ym) ** 2).sum()
    out = np.zeros((W, C))
    for ch in range(C):
        vals = A[:n, :, ch]                        # (n, W)
        s = ((y - ym)[:, None] * (vals - vals.mean(axis=0))).sum(axis=0) / denom
        s = fill_invalid(s, valid)
        out[:, ch] = np.clip(gauss1d(s, 60), -SLOPE_LIMIT, SLOPE_LIMIT)
    return out


def anchors(A, depth):
    """Colour of the real top edge per column, with blocked columns filled in."""
    W, C = A.shape[1], A.shape[2]
    valid = depth >= 3
    out = np.zeros((W, C))
    for ch in range(C):
        b = A[:3, :, ch].mean(axis=0)
        out[:, ch] = gauss1d(fill_invalid(b, valid), 4)
    return out


def bg_noise(A, depth):
    """High-frequency residual of a clean background patch."""
    W = A.shape[1]
    col = int(np.argmax(depth))                    # deepest clear column
    c0, c1 = max(0, col - 90), min(W, col + 90)
    rows = min(depth[c0:c1].min(), 60)
    patch = A[:max(rows, 12), c0:c1, :].astype(np.float64)
    res = []
    for ch in range(3):
        v = patch[:, :, ch]
        m = sum(np.roll(np.roll(v, dy, 0), dx, 1)
                for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0
        res.append((v - m)[2:-2, 2:-2].std())
    return float(np.mean(res))


def extend_top(A, n, depth):
    H, W, C = A.shape
    s = column_slopes(A, depth)
    b = anchors(A, depth)
    ys = np.arange(-n, 0, dtype=np.float64)[:, None]
    out = np.zeros((n, W, C))
    for ch in range(C):
        out[:, :, ch] = b[None, :, ch] + s[None, :, ch] * ys
    out += np.random.default_rng(7).normal(0, bg_noise(A, depth), out.shape)
    return np.concatenate([out, A], axis=0)


def extend_side(A, n, left):
    if n <= 0:
        return A
    H, W, C = A.shape
    win = 60
    y = np.arange(win, dtype=np.float64)
    ym = y.mean()
    denom = ((y - ym) ** 2).sum()
    out = np.zeros((H, n, C))
    for ch in range(C):
        vals = (A[:, :win, ch] if left else A[:, W - win:, ch]).T
        s = ((y - ym)[:, None] * (vals - vals.mean(axis=0))).sum(axis=0) / denom
        s = np.clip(gauss1d(s, 40), -SLOPE_LIMIT, SLOPE_LIMIT)
        edge = gauss1d((A[:, :3, ch] if left else A[:, W - 3:, ch]).mean(axis=1), 3)
        xs = (np.arange(-n, 0) if left else np.arange(1, n + 1))[None, :]
        out[:, :, ch] = edge[:, None] + s[:, None] * xs
    return np.concatenate([out, A] if left else [A, out], axis=1)


def subject_cols(A, depth):
    blocked = np.where(depth < A.shape[0])[0]
    return (int(blocked.min()), int(blocked.max())) if len(blocked) else (0, A.shape[1] - 1)


def process(src, dst, px=DEFAULT_PX, fit=True):
    im = Image.open(src).convert('RGB')
    A = np.asarray(im).astype(np.float64)
    H, W, _ = A.shape
    depth = background_depth(A)
    clear = int(depth.min())
    if clear < 3:
        raise SystemExit(f"{src}: the subject touches the top edge, nothing to sample")

    # the strip shrinks when the frame is resampled back, so oversize it first
    T = int(round(px * H / float(H - px))) if fit else px
    B = extend_top(A, T, depth)

    if fit:                                        # widen so W/H is unchanged
        side = int(round(T * W / float(H)))
        c0, c1 = subject_cols(A, depth)
        lm, rm = c0, W - c1
        left = int(round(side * lm / float(lm + rm))) if lm + rm else side // 2
        B = extend_side(extend_side(B, left, True), side - left, False)

    out = Image.fromarray(np.clip(np.rint(B), 0, 255).astype(np.uint8))
    if fit:
        out = out.resize((W, H), Image.LANCZOS)
    if dst.lower().endswith(('.jpg', '.jpeg')):
        out.save(dst, quality=97, subsampling=0)
    else:
        out.save(dst)

    V = np.asarray(Image.open(dst).convert('RGB')).astype(np.float64)
    print(f"{dst}: +{T}px strip, grain={bg_noise(A, depth):.2f}, {W}x{H} -> {out.size[0]}x{out.size[1]}"
          f" | clearance {clear}px -> {int(background_depth(V).min())}px")


if __name__ == '__main__':
    argv = sys.argv[1:]
    px, fit, args = DEFAULT_PX, True, []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--no-fit':
            fit = False
        elif a.startswith('--px='):
            px = int(a.split('=', 1)[1])
        elif a == '--px':
            i += 1
            px = int(argv[i])
        elif a.startswith('--'):
            raise SystemExit(f'unknown flag {a}')
        else:
            args.append(a)
        i += 1
    if len(args) != 2:
        raise SystemExit(__doc__)
    process(args[0], args[1], px=px, fit=fit)
