"""Rebuild a crown the frame cut off, using another frame of the same shoot.

Usage:  python3 scripts/transplant_crown.py TARGET REFERENCE OUT [--px 55]

The head outline is traced in both frames with a soft matte, a similarity
transform is fitted on the parts of the outline the cut did not touch, and the
reference crown is warped in above a blend line that sits inside the hair.
Tone is matched on the overlap band before compositing, then the top margin is
added the same way top_margin.py does it.
"""
import sys
import numpy as np
from PIL import Image
from gauss import gauss1d
import top_margin as tm

MATTE_K = 30.0          # contrast of the soft hair/background matte
DOME = 170              # outline rows that still belong to the crown, not falling hair
SCALE = (0.60, 1.85)    # head-size ratios the reference search covers
BLEND = 16              # rows of real hair the hand-over ramp is allowed to touch
FEATHER = 2.2           # px of softness on a lifted outline
LIFT_MAX = 10           # gap up to which the frame's own hair is stretched instead


def matte(A, rows=400):
    """Soft alpha of the subject against the studio background, top band only."""
    g = A[:rows].mean(axis=2)
    bg = gauss1d(np.median(A[:6].mean(axis=2), axis=0), 60)
    return np.clip((bg[None, :] - g) / MATTE_K, 0, 1)


def outline(a, thr=0.5):
    hit = a > thr
    return np.where(hit.any(0), hit.argmax(0), -1).astype(np.float64)


def fit_similarity(y_t, y_r, cols_t):
    """Grid search s, dx, dy mapping the reference dome onto the target's."""
    ok = np.where((y_r >= 0) & (y_r <= DOME))[0]
    best = None
    for s in np.arange(SCALE[0], SCALE[1], 0.002):
        for dx in range(-200, 201):
            x9 = (cols_t - dx) / s
            m = (x9 >= ok.min()) & (x9 <= ok.max())
            if m.sum() < 80:
                continue
            d = y_t[m] - s * np.interp(x9[m], ok, y_r[ok])
            dy = np.median(d)
            err = np.mean((d - dy) ** 2)
            if best is None or err < best[0]:
                best = (err, s, dx, dy)
    if best is None:
        raise SystemExit("could not match the reference head to the target")
    return best


def sample(A, mx, my):
    H, W = A.shape[:2]
    x0 = np.clip(np.floor(mx).astype(int), 0, W - 2)
    y0 = np.clip(np.floor(my).astype(int), 0, H - 2)
    fx = np.clip(mx - x0, 0, 1)[..., None]
    fy = np.clip(my - y0, 0, 1)[..., None]
    a = A[y0, x0] * (1 - fx) + A[y0, x0 + 1] * fx
    b = A[y0 + 1, x0] * (1 - fx) + A[y0 + 1, x0 + 1] * fx
    return a * (1 - fy) + b * fy


def curve_from(yr, s, dx, dy, W):
    """The reference outline expressed in the target's coordinates."""
    ok = np.where(yr >= 0)[0]
    x9 = (np.arange(W) - dx) / s
    inb = (x9 >= ok.min()) & (x9 <= ok.max())
    c = s * np.interp(np.clip(x9, ok.min(), ok.max()), ok, yr[ok]) + dy
    c = gauss1d(c, 6)          # the traced outline is ragged on wispy strands,
    return np.where(inb, c, np.inf)   # and a ragged curve combs the rebuilt edge


def lift_crown(B, N, yt, curve, cols, span=50):
    """Stretch the target's own hair up to the fitted outline.

    Only a few pixels are missing here, so borrowing texture from another frame
    costs more than it buys: this walks each column's top `span` rows upward by
    the gap, fading the shift out, which keeps the photo's own strands.
    """
    for xi in cols:
        g = yt[xi] - curve[xi]
        y0 = int(np.floor(curve[xi] + N))
        y1 = int(yt[xi] + N + span)
        col = B[:, xi, :].copy()
        ys = np.arange(y0, y1, dtype=np.float64)
        t = (ys - (curve[xi] + N)) / max(y1 - (curve[xi] + N), 1e-6)
        src = np.clip(ys + g * (1 - t), 0, B.shape[0] - 2)
        lo = np.floor(src).astype(int)
        f = (src - lo)[:, None]
        B[y0:y1, xi, :] = col[lo] * (1 - f) + col[lo + 1] * f
    return B


def run(tgt_path, ref_path, dst, px=tm.DEFAULT_PX):
    T = np.asarray(Image.open(tgt_path).convert('RGB')).astype(np.float64)
    R = np.asarray(Image.open(ref_path).convert('RGB')).astype(np.float64)
    H, W, _ = T.shape

    at, ar = matte(T), matte(R)
    yt, yr = outline(at), outline(ar)
    cut = np.where((yt >= 0) & (yt <= 1))[0]
    if not len(cut):
        raise SystemExit(f"{tgt_path}: nothing is cut, use top_margin.py")
    lo, hi = cut.min(), cut.max()

    # fit on the dome either side of the cut, where the target outline is intact
    sides = np.where((yt >= 4) & (yt <= DOME))[0]
    sides = sides[(sides < lo - 6) | (sides > hi + 6)]
    err, s, dx, dy = fit_similarity(yt[sides], yr, sides.astype(np.float64))
    print(f"  outline fit: scale={s:.3f} dx={dx} dy={dy:.1f} rms={np.sqrt(err):.2f}px")

    # The reference supplies the shape of the missing arc, not its position:
    # anchored to the target's own outline at both ends of the cut, so only the
    # cut columns are rebuilt and real hair either side is never moved.
    curve = curve_from(yr, s, dx, dy, W)
    span = np.arange(lo, hi + 1)
    yL, yR = np.median(yt[max(0, lo - 8):lo]), np.median(yt[hi + 1:hi + 9])
    cL, cR = curve[max(0, lo - 4)], curve[min(W - 1, hi + 4)]
    gap = np.zeros(W)
    if np.isfinite(cL) and np.isfinite(cR) and np.isfinite(curve[span]).all():
        t = (span - lo) / float(max(hi - lo, 1))
        arc = curve[span] + (yL - cL) * (1 - t) + (yR - cR) * t
        gap[span] = np.clip(yt[span] - arc, 0, None)
        gap = np.where(gauss1d(gap, 4) > 0.5, gauss1d(gap, 4), 0.0)
    lift = 0 < gap.max() <= LIFT_MAX

    apex = (s * yr[yr >= 0].min() + dy)
    crown = int(np.ceil(max(gap.max(), -apex, 0.0))) + 6
    Tm = px
    N = Tm + crown

    depth = np.maximum(tm.background_depth(T), 1)
    B = tm.extend_top(T, N, depth)                       # background above everything

    if lift:
        cols = np.where(gap > 0.3)[0]
        print(f"  lifting the frame's own hair by up to {gap.max():.1f}px over {len(cols)} columns")
        B = lift_crown(B, N, yt, curve, cols)
        alpha = np.clip(((np.arange(N + 60)[:, None] - (curve[None, :] + N)) / FEATHER) + 0.5, 0, 1)
        touched = np.zeros(W, bool)
        touched[cols] = True
        a = np.where(touched[None, :], alpha, 1.0)[:, :, None]
        bg = tm.extend_top(T, N, depth)[:N + 60]
        B[:N + 60] = B[:N + 60] * a + bg * (1 - a)

    else:
        # warp the reference into the extended target frame
        ys, xs = np.mgrid[0:N + 320, 0:W]
        mx = (xs - dx) / s
        my = (ys - N - dy) / s
        Rw = sample(R, mx, my)
        Aw = sample(ar[..., None], mx, my)[..., 0]
        inside = (mx >= 0) & (mx < W - 1) & (my >= 0) & (my < ar.shape[0] - 1)
        Aw = np.where(inside, Aw, 0.0)

        # tone match on hair both frames agree on
        band = slice(N + 30, N + 150)
        m = (Aw[band] > 0.95) & (at[30:150] > 0.95)
        if m.sum() > 2000:
            for ch in range(3):
                a_, b_ = Rw[band][..., ch][m], B[band][..., ch][m]
                g = b_.std() / max(a_.std(), 1e-6)
                Rw[..., ch] = (Rw[..., ch] - a_.mean()) * np.clip(g, 0.8, 1.25) + b_.mean()
            print(f"  tone matched on {int(m.sum())} px of shared hair")

        # vertical blend: the reference fills only what the frame cut off, and hands
        # back to the real hair a few rows in — a wide ramp washes out real texture
        y1, y2 = N - 1, N + BLEND
        w = np.clip((y2 - ys) / float(y2 - y1), 0, 1)
        a = np.clip(Aw, 0, 1) * w
        B[:N + 320] = Rw * a[..., None] + B[:N + 320] * (1 - a[..., None])

    side = int(round((H + N) * W / float(H))) - W
    left = side // 2                                  # keep the subject centred
    B = tm.extend_side(tm.extend_side(B, left, True), side - left, False)
    out = tm.save(B, dst)

    V = np.asarray(Image.open(dst).convert('RGB')).astype(np.float64)
    yv = outline(matte(V))
    print(f"{dst}: crown rebuilt {crown - 6}px, +{Tm}px margin, "
          f"clearance 0 -> {int(yv[yv >= 0].min())}px, {out.size[0]}x{out.size[1]}")


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
    if len(args) != 3:
        raise SystemExit(__doc__)
    run(args[0], args[1], args[2], px)
