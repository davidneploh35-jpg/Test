"""Re-tone a frame's studio backdrop to match a reference frame.

Usage:  python3 scripts/match_backdrop.py IN REFERENCE OUT

Both backdrops are modelled as a smooth field interpolated across the subject
from the clean margins, and the difference between the two fields is added to
the background only. The subject keeps its own pixels.
"""
import sys
import numpy as np
from PIL import Image
from gauss import gauss1d


def blur2d(a, s):
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        out[i] = gauss1d(a[i], s)
    for j in range(a.shape[1]):
        out[:, j] = gauss1d(out[:, j], s)
    return out


def field(A, margin=0.06):
    """Smooth backdrop estimate: read the clean margins, sweep across the frame."""
    H, W, C = A.shape
    m = max(8, int(W * margin))
    out = np.zeros_like(A)
    t = np.linspace(0, 1, W)[None, :]
    for ch in range(C):
        left = gauss1d(np.median(A[:, :m, ch], axis=1), 40)
        right = gauss1d(np.median(A[:, W - m:, ch], axis=1), 40)
        out[:, :, ch] = left[:, None] * (1 - t) + right[:, None] * t
    return out


def subject_alpha(A, f):
    """1 over the model, 0 over clean backdrop, feathered between.

    Built from colour rather than from brightness alone: the backdrop is
    neutral, so skin separates by its red bias while cloth and hair separate by
    how far they sit from the backdrop field. Each row is closed run by run, so
    the gap between the legs stays background.
    """
    H, W, _ = A.shape
    g = A.mean(axis=2)
    fg = f.mean(axis=2)
    skin = (A[:, :, 0] - A[:, :, 2] > 7) & (A[:, :, 0] > 110)
    cloth = g > fg + 18
    hair = g < fg - 35
    m = skin | cloth | hair
    out = np.zeros((H, W), bool)
    for r in range(H):                             # close each run, never bridge two
        on = np.where(m[r])[0]
        if not len(on):
            continue
        for part in np.split(on, np.where(np.diff(on) > 12)[0] + 1):
            if len(part) > 4:
                out[r, part.min():part.max() + 1] = True
    a = blur2d(out.astype(np.float64), 3.0)
    return np.clip(a * 1.25, 0, 1)[..., None]


def run(src, ref, dst):
    A = np.asarray(Image.open(src).convert('RGB')).astype(np.float64)
    R = np.asarray(Image.open(ref).convert('RGB')).astype(np.float64)
    H, W, _ = A.shape

    fa = field(A)
    fr = np.asarray(Image.fromarray(np.uint8(np.clip(field(R), 0, 255)))
                    .resize((W, H), Image.LANCZOS)).astype(np.float64)
    alpha = subject_alpha(A, fa)
    B = A + (fr - fa) * (1.0 - alpha)

    out = Image.fromarray(np.clip(np.rint(B), 0, 255).astype(np.uint8))
    out.save(dst, quality=98, subsampling=0) if dst.lower().endswith(('.jpg', '.jpeg')) \
        else out.save(dst)
    V = np.asarray(Image.open(dst).convert('RGB')).astype(np.float64)
    pts = {'верх-лево': (0.03, 0.08), 'верх-право': (0.03, 0.92),
           'серед-лево': (0.35, 0.06), 'низ-право': (0.80, 0.94)}
    for name, (fy, fx) in pts.items():
        y, x = int(H * fy), int(W * fx)
        a = A[y - 12:y + 12, x - 12:x + 12].mean()
        v = V[y - 12:y + 12, x - 12:x + 12].mean()
        ry, rx = int(R.shape[0] * fy), int(R.shape[1] * fx)
        r = R[ry - 12:ry + 12, rx - 12:rx + 12].mean()
        print(f"  {name:11s} {a:6.1f} -> {v:6.1f}   (эталон {r:.1f})")
    print(f"{dst}: {W}x{H}")


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    run(*sys.argv[1:])
