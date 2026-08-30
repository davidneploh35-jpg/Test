"""Bring a frame to a given amount of background above the head.

Usage:  python3 scripts/set_margin.py IN OUT [--px 55] [--ratio 0.75]

Frames shot tight get the strip synthesised above them, exactly as
top_margin.py does it. Frames shot loose are cropped down instead - no pixel is
invented and none is resampled - and the sides are cropped to match so the
aspect ratio survives.
"""
import sys
import numpy as np
from PIL import Image
import top_margin as tm
from transplant_crown import matte, outline


def clearance(A):
    y = outline(matte(A, rows=min(900, A.shape[0])))
    return int(y[y >= 0].min()) if (y >= 0).any() else 0


def process(src, dst, px=tm.DEFAULT_PX, ratio=None):
    im = Image.open(src).convert('RGB')
    A = np.asarray(im).astype(np.float64)
    H, W, _ = A.shape
    have = clearance(A)

    if have <= px:
        tm.process(src, dst, px=px - have, ratio=ratio or W / float(H))
        return

    top = have - px
    nh = H - top
    nw = int(round(nh * (ratio or W / float(H))))
    if nw > W:
        raise SystemExit(f"{src}: cropping to that ratio would need {nw - W}px more width")
    left = (W - nw) // 2
    out = im.crop((left, top, left + nw, H))
    out.save(dst, quality=98, subsampling=0) if dst.lower().endswith(('.jpg', '.jpeg')) \
        else out.save(dst)
    got = clearance(np.asarray(Image.open(dst).convert('RGB')).astype(np.float64))
    print(f"{dst}: cropped {top}px off the top and {W - nw}px of width, "
          f"{W}x{H} -> {out.size[0]}x{out.size[1]} | clearance {have}px -> {got}px")


if __name__ == '__main__':
    argv = sys.argv[1:]
    px, ratio, args = tm.DEFAULT_PX, None, []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--px':
            i += 1
            px = int(argv[i])
        elif a.startswith('--px='):
            px = int(a.split('=', 1)[1])
        elif a == '--ratio':
            i += 1
            ratio = float(argv[i])
        elif a.startswith('--ratio='):
            ratio = float(a.split('=', 1)[1])
        elif a.startswith('--'):
            raise SystemExit(f'unknown flag {a}')
        else:
            args.append(a)
        i += 1
    if len(args) != 2:
        raise SystemExit(__doc__)
    process(args[0], args[1], px, ratio)
