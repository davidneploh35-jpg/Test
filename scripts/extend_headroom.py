import numpy as np
from PIL import Image
from gauss import gauss1d

REF_FRAC = 81/1448.0   # headroom fraction measured on reference image
OUT_W, OUT_H = 1920, 2560

def fit_line(vals, coords):
    # least squares line fit: vals shape (n, cols)
    n = len(coords)
    c = coords.astype(np.float64)
    cm = c.mean(); vm = vals.mean(axis=0)
    denom = ((c-cm)**2).sum()
    slope = (((c-cm)[:,None] * (vals - vm)).sum(axis=0)) / denom
    intercept = vm - slope*cm
    return slope, intercept

def extend_left(A, n):
    if n <= 0: return A
    H, W, C = A.shape
    win = 60
    coords = np.arange(win)
    out = np.zeros((H, n, C), np.float64)
    for ch in range(C):
        vals = A[:, :win, ch].T                       # (win, H)
        s, _ = fit_line(vals, coords)                 # per row
        s = np.clip(gauss1d(s, 40), -0.12, 0.12)
        b = gauss1d(A[:, :3, ch].mean(axis=1), 3)     # anchor on the real edge column
        xs = np.arange(-n, 0)[None, :]                # (1, n)
        out[:, :, ch] = b[:, None] + s[:, None]*xs
    return np.concatenate([out, A], axis=1)

def extend_right(A, n):
    if n <= 0: return A
    H, W, C = A.shape
    win = 60
    coords = np.arange(W-win, W)
    out = np.zeros((H, n, C), np.float64)
    for ch in range(C):
        vals = A[:, W-win:, ch].T
        s, _ = fit_line(vals, coords)
        s = np.clip(gauss1d(s, 40), -0.12, 0.12)
        b = gauss1d(A[:, W-3:, ch].mean(axis=1), 3)   # anchor on the real edge column
        xs = (np.arange(1, n+1))[None, :]
        out[:, :, ch] = b[:, None] + s[:, None]*xs
    return np.concatenate([A, out], axis=1)

def extend_top(A, n, noise_sigma):
    H, W, C = A.shape
    win = 62                                          # rows 0..61 are pure background
    coords = np.arange(win)
    out = np.zeros((n, W, C), np.float64)
    rng = np.random.default_rng(7)
    for ch in range(C):
        vals = A[:win, :, ch]                         # (win, W)
        s, _ = fit_line(vals, coords)                 # per column
        s = np.clip(gauss1d(s, 80), -0.12, 0.12)
        b = gauss1d(A[:3, :, ch].mean(axis=0), 4)     # anchor on the real top row
        ys = np.arange(-n, 0)[:, None]                # (n,1)
        out[:, :, ch] = b[None, :] + s[None, :]*ys
    out += rng.normal(0, noise_sigma, out.shape)
    return np.concatenate([out, A], axis=0)

def hair_top(A):
    g = A.mean(axis=2)
    bg = g[:5].mean()
    dark = g < (bg - 25)
    rows = np.where(dark.sum(axis=1) > 3)[0]
    return int(rows[0])

def subject_cols(A):
    from PIL import ImageFilter
    im = Image.fromarray(A.astype(np.uint8)).convert('L').filter(ImageFilter.GaussianBlur(1.5))
    a = np.asarray(im).astype(np.float32)
    gy, gx = np.gradient(a)
    strong = np.hypot(gx, gy) > 6
    cols = np.where(strong.sum(axis=0) > 5)[0]
    return int(cols.min()), int(cols.max())

def bg_noise(A):
    # high-frequency residual std in a clean background patch
    patch = A[10:60, 10:200, :].astype(np.float64)
    sm = np.zeros_like(patch)
    for ch in range(3):
        v = patch[:, :, ch]
        sm[:, :, ch] = (v[:-0 or None] if False else v)
    # simple 3x3 mean
    k = np.ones((3,3))/9.0
    res = []
    for ch in range(3):
        v = patch[:, :, ch]
        m = np.zeros_like(v)
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                m += np.roll(np.roll(v, dy, 0), dx, 1)
        m /= 9.0
        res.append((v-m)[2:-2, 2:-2].std())
    return float(np.mean(res))

def process(src, dst):
    im = Image.open(src).convert('RGB')
    A = np.asarray(im).astype(np.float64)
    H, W, _ = A.shape
    top = hair_top(A)
    T = int(round((REF_FRAC*H - top) / (1.0 - REF_FRAC)))
    newH = H + T
    newW = int(round(newH * 0.75))
    side = newW - W
    c0, c1 = subject_cols(A)
    lm, rm = c0, W - c1
    left = int(round(side * lm / float(lm + rm)))
    right = side - left
    sigma = bg_noise(A)
    B = extend_left(A, left)
    B = extend_right(B, right)
    B = extend_top(B, T, sigma)
    B = np.clip(np.rint(B), 0, 255)
    out = Image.fromarray(B.astype(np.uint8)).resize((OUT_W, OUT_H), Image.LANCZOS)
    out.save(dst, quality=97, subsampling=0)
    # verification
    V = np.asarray(out).astype(np.float64)
    nt = hair_top(V)
    print(f"{dst}: added top={T}px left={left} right={right} noise={sigma:.2f} "
          f"canvas={B.shape[1]}x{B.shape[0]} -> {OUT_W}x{OUT_H} | headroom {top/H:.4f} -> {nt/OUT_H:.4f} (ref {REF_FRAC:.4f})")


if __name__ == '__main__':
    import sys
    process(sys.argv[1], sys.argv[2])
