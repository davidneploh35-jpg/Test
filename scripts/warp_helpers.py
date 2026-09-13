import numpy as np
from PIL import Image
from gauss import gauss1d
from hairfix import ALIGN, fit_affine, bg_model, matte, blur2d, warp

def tps_fit(src, dst, lam=1.0):
    """TPS mapping dst -> src (inverse map for warping), with regularisation lam"""
    n = len(dst)
    d = np.linalg.norm(dst[:, None, :] - dst[None, :, :], axis=2)
    K = np.where(d > 0, d*d*np.log(np.maximum(d, 1e-9)), 0.0)
    K = K + lam*np.eye(n)
    P = np.c_[np.ones(n), dst]
    L = np.zeros((n+3, n+3))
    L[:n, :n] = K; L[:n, n:] = P; L[n:, :n] = P.T
    Y = np.zeros((n+3, 2)); Y[:n] = src
    W = np.linalg.solve(L, Y)
    return W, dst

def tps_apply(W, ctrl, X, Y):
    """evaluate TPS at grid coords X, Y (1-D arrays broadcast) -> source coords"""
    n = len(ctrl)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    out = np.zeros((len(pts), 2))
    chunk = 400000
    for i in range(0, len(pts), chunk):
        p = pts[i:i+chunk]
        d = np.linalg.norm(p[:, None, :] - ctrl[None, :, :], axis=2)
        U = np.where(d > 0, d*d*np.log(np.maximum(d, 1e-9)), 0.0)
        A = np.c_[U, np.ones(len(p)), p]
        out[i:i+chunk] = A @ W
    return out[:, 0].reshape(X.shape), out[:, 1].reshape(X.shape)

def sample(arr, mx, my):
    """bilinear sample arr (HxWxC or HxW) at float coords"""
    single = arr.ndim == 2
    a = arr[:, :, None] if single else arr
    H, W, C = a.shape
    x0 = np.floor(mx).astype(np.int64); y0 = np.floor(my).astype(np.int64)
    fx = (mx - x0)[..., None]; fy = (my - y0)[..., None]
    x0c = np.clip(x0, 0, W-2); y0c = np.clip(y0, 0, H-2)
    p00 = a[y0c, x0c]; p10 = a[y0c, x0c+1]; p01 = a[y0c+1, x0c]; p11 = a[y0c+1, x0c+1]
    out = (p00*(1-fx)*(1-fy) + p10*fx*(1-fy) + p01*(1-fx)*fy + p11*fx*fy)
    inside = (mx >= 0) & (mx <= W-1) & (my >= 0) & (my <= H-1)
    out = out * inside[..., None]
    return out[:, :, 0] if single else out

def hf_energy(a, mask):
    g = a.mean(axis=2)
    lap = np.abs(4*g[1:-1,1:-1] - g[:-2,1:-1] - g[2:,1:-1] - g[1:-1,:-2] - g[1:-1,2:])
    m = mask[1:-1,1:-1] > 0.6
    return lap[m].mean() if m.sum() else 0.0

def match_sharpness(img, tgt, mask, tol=0.03):
    """blur img until its high-frequency energy matches tgt inside mask"""
    e_t = hf_energy(tgt, mask)
    best, best_s = img, 0.0
    for s in np.arange(0.0, 3.01, 0.25):
        cand = img if s == 0 else np.stack([blur2d(img[:,:,c], s) for c in range(3)], axis=2)
        e = hf_energy(cand, mask)
        if e <= e_t*(1+tol):
            return cand, s, e, e_t
        best, best_s = cand, s
    return best, best_s, hf_energy(best, mask), e_t
