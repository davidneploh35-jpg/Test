import numpy as np
from PIL import Image, ImageFilter
from gauss import gauss1d
import extend_mod as E

REF_FRAC = 81/1448.0
OUT_W, OUT_H = 1920, 2560
SRC = '/home/user/Test/photos/photo_5_haircolor.png'
ORIG = '/root/.claude/uploads/92ed6dba-dbfa-5416-bab6-95a063ee99a3/78db79cd-image.jpg'

def extend_top(A, n, noise_sigma, win):
    H, W, C = A.shape
    coords = np.arange(win)
    out = np.zeros((n, W, C), np.float64)
    rng = np.random.default_rng(11)
    for ch in range(C):
        s, _ = E.fit_line(A[:win, :, ch], coords)
        s = np.clip(gauss1d(s, 40), -0.12, 0.12)
        b = gauss1d(A[:3, :, ch].mean(axis=0), 4)
        ys = np.arange(-n, 0)[:, None]
        out[:, :, ch] = b[None, :] + s[None, :]*ys
    out += rng.normal(0, noise_sigma, out.shape)
    return np.concatenate([out, A], axis=0)

def run(sharpen, dst):
    A = np.asarray(Image.open(SRC).convert('RGB')).astype(np.float64)
    O = np.asarray(Image.open(ORIG).convert('RGB')).astype(np.float64)
    H, W, _ = A.shape
    top = E.hair_top(O)                      # silhouette is unchanged by the recolour
    T = int(round((REF_FRAC*H - top)/(1-REF_FRAC)))
    newH = H + T; newW = int(round(newH*0.75)); side = newW - W
    c0, c1 = E.subject_cols(A)
    left = int(round(side * c0/float(c0 + (W-c1)))); right = side - left
    sigma = E.bg_noise(A)
    win = min(30, top-6)
    B = E.extend_left(A, left); B = E.extend_right(B, right)
    B = extend_top(B, T, sigma, win)
    B = np.clip(np.rint(B), 0, 255).astype(np.uint8)
    im = Image.fromarray(B).resize((OUT_W, OUT_H), Image.LANCZOS)
    if sharpen:
        im = im.filter(ImageFilter.UnsharpMask(radius=1.6, percent=45, threshold=2))
    im.save(dst) if dst.endswith('.png') else im.save(dst, quality=97, subsampling=0)
    V = np.asarray(im.convert('RGB')).astype(np.float64)
    nt = E.hair_top(V)
    print(f'{dst}: top+{T} sides {left}/{right} win={win} canvas {B.shape[1]}x{B.shape[0]} -> {OUT_W}x{OUT_H} '
          f'| headroom {top/H:.4f} -> {nt/OUT_H:.4f} (ref {REF_FRAC:.4f}) | upscale x{OUT_H/(H+T):.3f}')
    return im

if __name__ == '__main__':
    run(False, 'p5_plain.png')
    run(True,  'p5_sharp.png')
