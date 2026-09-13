"""Erase a small invented detail from smooth fabric.

Usage:  python3 scripts/erase_detail.py IN OUT x0,y0,x1,y1 [more boxes...]

Each box is refilled by interpolating vertically between the fabric just above
and just below it, so seams and folds running down through the box survive.
Grain is matched to the surrounding cloth so the patch does not read as
plastic.
"""
import sys
import numpy as np
from PIL import Image
from gauss import gauss1d

REACH = 14          # rows of real fabric sampled either side
FEATHER = 3         # px of blend around the patch


def grain_of(A, box, reach=REACH):
    x0, y0, x1, y1 = box
    p = A[max(0, y0 - reach):y0, x0:x1, :]
    if p.size == 0:
        return 0.0
    k = sum(np.roll(np.roll(p, dy, 0), dx, 1)
            for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0
    return float((p - k)[1:-1, 1:-1].std())


def erase(A, box, rng):
    x0, y0, x1, y1 = box
    top = A[max(0, y0 - REACH):y0, x0:x1, :].mean(axis=0)
    bot = A[y1:y1 + REACH, x0:x1, :].mean(axis=0)
    n = y1 - y0
    t = (np.arange(n) + 0.5)[:, None, None] / n
    fill = top[None, :, :] * (1 - t) + bot[None, :, :] * t
    fill = fill + rng.normal(0, max(grain_of(A, box), 0.3), fill.shape)

    patch = A[y0:y1, x0:x1, :].copy()
    ys = np.clip(np.minimum(np.arange(n) + 1, n - np.arange(n)) / FEATHER, 0, 1)[:, None]
    xs = np.clip(np.minimum(np.arange(x1 - x0) + 1,
                            (x1 - x0) - np.arange(x1 - x0)) / FEATHER, 0, 1)[None, :]
    w = (ys * xs)[..., None]
    A[y0:y1, x0:x1, :] = fill * w + patch * (1 - w)
    return A


if __name__ == '__main__':
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    A = np.asarray(Image.open(src).convert('RGB')).astype(np.float64)
    rng = np.random.default_rng(11)
    for spec in sys.argv[3:]:
        box = tuple(int(v) for v in spec.split(','))
        A = erase(A, box, rng)
        print(f"  erased {box[2]-box[0]}x{box[3]-box[1]}px at ({box[0]}, {box[1]})")
    out = Image.fromarray(np.clip(np.rint(A), 0, 255).astype(np.uint8))
    out.save(dst, quality=98, subsampling=0) if dst.lower().endswith(('.jpg', '.jpeg')) \
        else out.save(dst)
    print(f"{dst}: {out.size[0]}x{out.size[1]}")
