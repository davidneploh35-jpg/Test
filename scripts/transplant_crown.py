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
    for s in np.arange(0.85, 1.251, 0.002):
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

    apex = (s * yr[yr >= 0].min() + dy)
    crown = int(np.ceil(max(0.0, -apex))) + 6
    Tm = int(round(px * H / float(H - px)))
    N = Tm + crown

    depth = np.maximum(tm.background_depth(T), 1)
    B = tm.extend_top(T, N, depth)                       # background above everything

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

    # vertical blend: reference above the cut, real hair below it
    y1, y2 = N - 2, N + 90                               # ramp lives inside the hair
    w = np.clip((y2 - ys) / float(y2 - y1), 0, 1)
    a = np.clip(Aw, 0, 1) * w
    B[:N + 320] = Rw * a[..., None] + B[:N + 320] * (1 - a[..., None])

    side = int(round(N * W / float(H)))
    c0, c1 = tm.subject_cols(T, depth)
    lm, rm = c0, W - c1
    left = int(round(side * lm / float(lm + rm))) if lm + rm else side // 2
    B = tm.extend_side(tm.extend_side(B, left, True), side - left, False)

    out = Image.fromarray(np.clip(np.rint(B), 0, 255).astype(np.uint8))
    out = out.resize((W, int(round(out.size[1] * W / out.size[0]))), Image.LANCZOS)
    out = out.crop((0, out.size[1] - H, W, out.size[1]))
    out.save(dst, quality=97, subsampling=0) if dst.lower().endswith(('.jpg', '.jpeg')) \
        else out.save(dst)

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
