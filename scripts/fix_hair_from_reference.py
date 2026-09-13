import numpy as np
from PIL import Image
from gauss import gauss1d

REF = '/root/.claude/uploads/92ed6dba-dbfa-5416-bab6-95a063ee99a3/3ba9081d-image.png'
TGT = {'i2':'/root/.claude/uploads/92ed6dba-dbfa-5416-bab6-95a063ee99a3/4dc6ba23-image.jpg',
       'i3':'/root/.claude/uploads/92ed6dba-dbfa-5416-bab6-95a063ee99a3/977293ce-image.jpg'}

# mediapipe indices used for alignment: upper face oval, brows, eyes, nose bridge
ALIGN = [10,338,297,332,284,251,389,356,454,323,          # right side / top
         109,67,103,54,21,162,127,234,93,                  # left side / top
         70,63,105,66,107,336,296,334,293,300,             # brows
         33,133,362,263,168,6,197,195,                     # eyes + nose bridge
         143,111,117,118,119,372,340,346,347,348]          # upper cheeks

def blur2d(a, sigma):
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        out[i] = gauss1d(a[i], sigma)
    out2 = np.empty_like(out)
    for j in range(a.shape[1]):
        out2[:, j] = gauss1d(out[:, j], sigma)
    return out2

def fit_affine(src, dst):
    """least-squares affine mapping src -> dst"""
    n = len(src)
    A = np.zeros((2*n, 6)); b = np.zeros(2*n)
    A[0::2, 0] = src[:,0]; A[0::2, 1] = src[:,1]; A[0::2, 2] = 1; b[0::2] = dst[:,0]
    A[1::2, 3] = src[:,0]; A[1::2, 4] = src[:,1]; A[1::2, 5] = 1; b[1::2] = dst[:,1]
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    M = np.array([[p[0],p[1],p[2]],[p[3],p[4],p[5]],[0,0,1]])
    res = (np.c_[src, np.ones(n)] @ M.T)[:, :2] - dst
    return M, np.linalg.norm(res, axis=1)

def bg_model(a, head_lo, head_hi):
    """smooth background estimate: interpolate horizontally across the head band"""
    g = a.copy()
    H, W, C = a.shape
    est = np.zeros_like(a)
    lo = max(0, head_lo - 40); hi = min(W, head_hi + 40)
    for ch in range(C):
        v = g[:, :, ch]
        left = np.median(v[:, max(0,lo-120):lo], axis=1)
        right = np.median(v[:, hi:hi+120], axis=1)
        left = gauss1d(left, 25); right = gauss1d(right, 25)
        t = np.linspace(0, 1, hi-lo)[None, :]
        band = left[:, None]*(1-t) + right[:, None]*t
        e = v.copy(); e[:, lo:hi] = band
        est[:, :, ch] = e
    return est

def matte(a, bg, k=28.0):
    """soft alpha: how much darker than the local background"""
    d = (bg.mean(axis=2) - a.mean(axis=2)) / k
    return np.clip(d, 0, 1)

def warp(arr, Minv, size):
    """arr: HxWxC float or HxW float; map target->ref with Minv"""
    c = (Minv[0,0], Minv[0,1], Minv[0,2], Minv[1,0], Minv[1,1], Minv[1,2])
    if arr.ndim == 2:
        im = Image.fromarray(arr.astype(np.float32), 'F')
        return np.asarray(im.transform(size, Image.AFFINE, c, Image.BICUBIC)).astype(np.float64)
    out = []
    for ch in range(arr.shape[2]):
        im = Image.fromarray(arr[:,:,ch].astype(np.float32), 'F')
        out.append(np.asarray(im.transform(size, Image.AFFINE, c, Image.BICUBIC)).astype(np.float64))
    return np.stack(out, axis=2)

def process(key, out_path, dbg=True):
    ref = np.asarray(Image.open(REF).convert('RGB')).astype(np.float64)
    tgt = np.asarray(Image.open(TGT[key]).convert('RGB')).astype(np.float64)
    lr = np.load('lm_ref.npy'); lt = np.load(f'lm_{key}.npy')
    M, res = fit_affine(lr[ALIGN], lt[ALIGN])
    iod = np.linalg.norm(lt[33]-lt[263])
    print(f'{key}: affine residual mean={res.mean():.2f}px max={res.max():.2f}px (IOD {iod:.0f})')

    Hr, Wr, _ = ref.shape; Ht, Wt, _ = tgt.shape
    # backgrounds + mattes
    ref_head = (int(lr[:,0].min()-1.2*np.linalg.norm(lr[33]-lr[263])), int(lr[:,0].max()+1.2*np.linalg.norm(lr[33]-lr[263])))
    tgt_head = (int(lt[:,0].min()-1.2*iod), int(lt[:,0].max()+1.2*iod))
    bgr = bg_model(ref, *ref_head); bgt = bg_model(tgt, *tgt_head)
    ar = matte(ref, bgr); at = matte(tgt, bgt)

    Minv = np.linalg.inv(M)
    wref = warp(ref, Minv, (Wt, Ht))
    wa   = np.clip(warp(ar, Minv, (Wt, Ht)), 0, 1)

    # crown region mask in target coords
    yh = lt[10,1]                      # forehead top (hairline centre)
    y1 = yh - 0.10*iod                 # fully replaced above this
    y2 = yh + 0.75*iod                 # fully original below this
    ys = np.arange(Ht)[:, None]
    m = np.clip((y2 - ys) / (y2 - y1), 0, 1) * np.ones((1, Wt))
    xl = lt[:,0].min() - 1.4*iod; xr = lt[:,0].max() + 1.4*iod
    xs = np.arange(Wt)[None, :]
    m *= np.clip((xs - xl)/(0.35*iod), 0, 1) * np.clip((xr - xs)/(0.35*iod), 0, 1)
    m = blur2d(m, 12)

    # colour match warped ref hair to target hair in the overlap
    sel = (wa > 0.85) & (at > 0.85) & (m > 0.4)
    print('   overlap px', int(sel.sum()))
    wref_c = wref.copy()
    for ch in range(3):
        s = wref[:,:,ch][sel]; t = tgt[:,:,ch][sel]
        sd = s.std() if s.std() > 1e-6 else 1.0
        wref_c[:,:,ch] = (wref[:,:,ch] - s.mean()) * (t.std()/sd) + t.mean()

    content = wref_c * wa[:,:,None] + bgt * (1 - wa[:,:,None])
    out = tgt * (1 - m[:,:,None]) + content * m[:,:,None]
    out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    Image.fromarray(out).save(out_path, quality=97, subsampling=0)
    if dbg:
        x0 = int(lt[:,0].min()-1.0*iod); x1 = int(lt[:,0].max()+1.0*iod)
        y0 = 0; y1c = int(lt[152,1])
        Image.fromarray(out).crop((x0,y0,x1,y1c)).save(f'dbg_{key}_new.png')
        Image.fromarray(tgt.astype(np.uint8)).crop((x0,y0,x1,y1c)).save(f'dbg_{key}_old.png')
    return out

if __name__ == '__main__':
    process('i2', 'hairfix_i2.jpg')
    process('i3', 'hairfix_i3.jpg')
