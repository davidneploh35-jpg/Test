"""Bring a frame to a given amount of background above the head.

Usage:  python3 scripts/set_margin.py IN OUT [--px 55] [--ratio 0.75] [--size WxH]

Frames shot tight get the strip synthesised above them, exactly as
top_margin.py does it. Frames shot loose are cropped down instead - no pixel is
invented and none is resampled - and the sides are cropped to match so the
aspect ratio survives.

--size asks for exact delivered dimensions. The frame is scaled once, before
the crop, so the picture goes through a single resample rather than two.
"""
import sys
import numpy as np
from PIL import Image
import top_margin as tm
from gauss import gauss1d
from transplant_crown import matte, outline

SHARP_R = 1.1        # radius of the detail layer, px
SHARP_A = 0.24       # tuned so an enlargement lands back at the source detail,
                     # not past it: overshooting reads as crunch, not sharpness
SHARP_FLOOR = 2.0    # detail below this is grain, not structure
SHARP_CAP = 14.0     # ceiling on the correction, keeps edges from ringing


def blur2d(a, s):
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        out[i] = gauss1d(a[i], s)
    for j in range(a.shape[1]):
        out[:, j] = gauss1d(out[:, j], s)
    return out


def resharpen(A):
    """Put back the edge detail an enlargement costs, without touching the backdrop.

    The correction is gated on how much local detail there is, so grain in a
    smooth studio background is left alone, and capped so high-contrast edges
    like the hairline do not ring.
    """
    out = np.empty_like(A)
    for ch in range(A.shape[2]):
        v = A[:, :, ch]
        d = v - blur2d(v, SHARP_R)
        gate = np.clip((np.abs(d) - SHARP_FLOOR) / SHARP_FLOOR, 0, 1)
        out[:, :, ch] = v + np.clip(d * SHARP_A * gate, -SHARP_CAP, SHARP_CAP)
    return np.clip(out, 0, 255)


def clearance(A, run=12):
    """Rows of background above the subject.

    Takes the first row carrying a `run` of adjacent subject pixels rather than
    the first stray one: a speck of floor dirt or a mark on the backdrop at the
    frame edge is not the top of anybody's head.
    """
    rows = min(int(A.shape[0] * 0.4), 1400)
    a = matte(A, rows=rows) > 0.5
    for r in range(a.shape[0]):
        on = np.where(a[r])[0]
        if len(on) >= run:
            for part in np.split(on, np.where(np.diff(on) > 1)[0] + 1):
                if len(part) >= run:
                    return r
    return 0


def process(src, dst, px=tm.DEFAULT_PX, ratio=None, size=None):
    im = Image.open(src).convert('RGB')
    A = np.asarray(im).astype(np.float64)
    H, W, _ = A.shape
    have = clearance(A)

    if size:
        # solve for the scale that lands on `size` once `have` is cropped to `px`
        s = (size[1] - px) / float(H - have)
        im = im.resize((int(round(W * s)), int(round(H * s))), Image.LANCZOS)
        A = np.asarray(im).astype(np.float64)
        if s > 1.02:                      # an enlargement softens; give the edges back
            A = resharpen(A)
            im = Image.fromarray(np.rint(A).astype(np.uint8))
        H, W, _ = A.shape
        have = clearance(A)
        ratio = size[0] / float(size[1])
        print(f"  scaled by {s:.4f} to {W}x{H} before cropping")

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
    px, ratio, size, args = tm.DEFAULT_PX, None, None, []
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
        elif a == '--size':
            i += 1
            size = tuple(int(v) for v in argv[i].lower().split('x'))
        elif a.startswith('--size='):
            size = tuple(int(v) for v in a.split('=', 1)[1].lower().split('x'))
        elif a.startswith('--'):
            raise SystemExit(f'unknown flag {a}')
        else:
            args.append(a)
        i += 1
    if len(args) != 2:
        raise SystemExit(__doc__)
    process(args[0], args[1], px, ratio, size)
