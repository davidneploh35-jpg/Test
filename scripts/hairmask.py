import numpy as np
from PIL import Image, ImageDraw
from hairfix import bg_model, matte, blur2d

BROWS = [70,63,105,66,107,336,296,334,293,300]

def convex_hull(P):
    P = sorted(map(tuple, P))
    def half(pts):
        h = []
        for p in pts:
            while len(h) >= 2 and ((h[-1][0]-h[-2][0])*(p[1]-h[-2][1]) - (h[-1][1]-h[-2][1])*(p[0]-h[-2][0])) <= 0:
                h.pop()
            h.append(p)
        return h
    return half(P)[:-1] + half(P[::-1])[:-1]

def _filt(a, r, op):
    out = a.copy()
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            if dx*dx+dy*dy > r*r: continue
            out = op(out, np.roll(np.roll(a, dy, 0), dx, 1))
    return out

def close(m, r):
    d = _filt(m, r, np.maximum)
    return _filt(d, r, np.minimum)

def hair_mask(a, lt, l1=100.0, l2=145.0, warm=6.0, feather=2.5):
    """soft hair mask: dark warm pixels in the head region, face interior excluded"""
    H,W,_ = a.shape
    iod = np.linalg.norm(lt[33]-lt[263]); yh = lt[10,1]
    bg = bg_model(a, int(lt[:,0].min()-1.3*iod), int(lt[:,0].max()+1.3*iod))
    alpha = matte(a, bg, k=30.0)
    L = a.mean(axis=2)
    col = np.clip((l2 - L)/(l2-l1), 0, 1) * np.clip((a[:,:,0]-a[:,:,2]-warm/2)/warm, 0, 1)
    ys = np.arange(H)[:,None]; xs = np.arange(W)[None,:]
    space = (np.clip((ys-(yh-1.9*iod))/(0.3*iod),0,1) * np.clip((lt[152,1]-ys)/(0.35*iod),0,1)
             * np.clip((xs-(lt[:,0].min()-1.6*iod))/(0.3*iod),0,1)
             * np.clip(((lt[:,0].max()+1.6*iod)-xs)/(0.3*iod),0,1))
    # exclude the face interior from the brows down (protects brows, eyes, skin)
    poly = Image.new('F', (W,H), 1.0)
    d = ImageDraw.Draw(poly)
    brow_y = lt[BROWS, 1].min()
    face_pts = lt[lt[:,1] >= brow_y - 0.15*iod]
    d.polygon(convex_hull(face_pts), fill=0.0)
    face = np.asarray(poly).astype(np.float64)
    face = blur2d(face, 3)
    m = col * space * face * np.clip(alpha*1.4, 0, 1)
    m = close(m, 6)                      # fill highlight holes inside the hair mass
    m = m * space * face
    return blur2d(m, feather), alpha, bg
