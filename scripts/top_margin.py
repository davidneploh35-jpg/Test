"""Add a small strip of background above the subject.

Usage:  python3 scripts/top_margin.py IN OUT [--px 55] [--no-fit]

The strip is synthesised from the existing top rows (per-column colour trend
plus matched grain), the sides are widened by the same proportion so the frame
keeps its aspect ratio, and the result is resampled back to the input size.
Pass --no-fit to keep the enlarged canvas instead.
"""
import sys
import numpy as np
from PIL import Image
from extend_headroom import extend_left, extend_right, extend_top, hair_top, subject_cols, bg_noise

DEFAULT_PX = 55


def process(src, dst, px=DEFAULT_PX, fit=True):
    im = Image.open(src).convert('RGB')
    A = np.asarray(im).astype(np.float64)
    H, W, _ = A.shape

    top = hair_top(A)                      # first row that holds the subject
    if top < 8:
        raise SystemExit(f"{src}: only {top}px of background above the subject — "
                         "nothing to sample, extend by hand")

    sigma = bg_noise(A)
    B = extend_top(A, px, sigma)

    if fit:                                # widen so W/H is unchanged
        side = int(round(px * W / float(H)))
        c0, c1 = subject_cols(A)
        lm, rm = c0, W - c1
        left = int(round(side * lm / float(lm + rm))) if lm + rm else side // 2
        B = extend_right(extend_left(B, left), side - left)

    B = np.clip(np.rint(B), 0, 255).astype(np.uint8)
    out = Image.fromarray(B)
    if fit:
        out = out.resize((W, H), Image.LANCZOS)
    out.save(dst, quality=97, subsampling=0) if dst.lower().endswith(('.jpg', '.jpeg')) else out.save(dst)

    V = np.asarray(Image.open(dst).convert('RGB')).astype(np.float64)
    print(f"{dst}: +{px}px top, grain={sigma:.2f}, {W}x{H} -> {out.size[0]}x{out.size[1]} | "
          f"clearance {top}px -> {hair_top(V)}px")


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
