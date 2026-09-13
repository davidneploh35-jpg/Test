import numpy as np
from PIL import Image
from hairfix import ALIGN, fit_affine, bg_model, matte, blur2d, warp
import hairfix2 as h2

TGT_PATH='/root/.claude/uploads/92ed6dba-dbfa-5416-bab6-95a063ee99a3/59e25ca1-image.jpg'
SRC_PATH='ref_mirrored.png'

def run(mode='affine', lam=200.0, y1f=-0.10, y2f=0.75, out='fix4.png', dbg=True):
    tgt=np.asarray(Image.open(TGT_PATH).convert('RGB')).astype(np.float64)
    src=np.asarray(Image.open(SRC_PATH).convert('RGB')).astype(np.float64)
    ls=np.load('lm_refm.npy'); lt=np.load('lm_i4.npy')
    Ht,Wt,_=tgt.shape; Hs,Ws,_=src.shape
    iod=np.linalg.norm(lt[33]-lt[263])
    M,res=fit_affine(ls[ALIGN], lt[ALIGN])
    print(f'{mode}: affine residual mean {res.mean():.2f} max {res.max():.2f}')

    iods=np.linalg.norm(ls[33]-ls[263])
    bgs=bg_model(src, int(ls[:,0].min()-1.2*iods), int(ls[:,0].max()+1.2*iods))
    bgt=bg_model(tgt, int(lt[:,0].min()-1.2*iod), int(lt[:,0].max()+1.2*iod))
    a_s=matte(src,bgs); a_t=matte(tgt,bgt)

    Minv=np.linalg.inv(M)
    if mode=='affine':
        wsrc=warp(src,Minv,(Wt,Ht)); wa=np.clip(warp(a_s,Minv,(Wt,Ht)),0,1)
    else:
        W_,ctrl=h2.tps_fit(ls[ALIGN], lt[ALIGN], lam=lam)
        ys=np.arange(Ht); xs=np.arange(Wt)
        X,Y=np.meshgrid(xs,ys)
        mx,my=h2.tps_apply(W_,ctrl,X,Y)
        # affine inverse map
        ax=Minv[0,0]*X+Minv[0,1]*Y+Minv[0,2]; ay=Minv[1,0]*X+Minv[1,1]*Y+Minv[1,2]
        # blend: TPS near/below face top, affine well above it
        ytop=lt[10,1]
        w=np.clip((Y-(ytop-1.3*iod))/(1.0*iod),0,1)[:, :]
        mx=w*mx+(1-w)*ax; my=w*my+(1-w)*ay
        wsrc=h2.sample(src,mx,my); wa=np.clip(h2.sample(a_s,mx,my),0,1)

    yh=lt[10,1]
    y1=yh+y1f*iod; y2=yh+y2f*iod
    ys=np.arange(Ht)[:,None]
    m=np.clip((y2-ys)/(y2-y1),0,1)*np.ones((1,Wt))
    xl=lt[:,0].min()-1.4*iod; xr=lt[:,0].max()+1.4*iod
    xs=np.arange(Wt)[None,:]
    m*=np.clip((xs-xl)/(0.35*iod),0,1)*np.clip((xr-xs)/(0.35*iod),0,1)
    m=blur2d(m,12)

    sel=(wa>0.85)&(a_t>0.85)&(m>0.4)
    print('   overlap px',int(sel.sum()))
    wc=wsrc.copy()
    for ch in range(3):
        s=wsrc[:,:,ch][sel]; t=tgt[:,:,ch][sel]
        wc[:,:,ch]=(wsrc[:,:,ch]-s.mean())*(t.std()/max(s.std(),1e-6))+t.mean()
    hm=((wa>0.7)&(m>0.7)).astype(float)
    wc,sig,e,et=h2.match_sharpness(wc,tgt,hm)
    print(f'   sharpness match: blur {sig:.2f}px  hf {e:.2f} vs target {et:.2f}')

    content=wc*wa[:,:,None]+bgt*(1-wa[:,:,None])
    o=tgt*(1-m[:,:,None])+content*m[:,:,None]
    o=np.clip(np.rint(o),0,255).astype(np.uint8)
    Image.fromarray(o).save(out)
    if dbg:
        x0=int(lt[:,0].min()-1.1*iod); x1=int(lt[:,0].max()+1.1*iod); y1c=int(lt[152,1]+0.3*iod)
        Image.fromarray(o).crop((x0,40,x1,y1c)).save(out.replace('.png','_head.png'))
    return o

if __name__=='__main__':
    import sys
    run(mode=sys.argv[1] if len(sys.argv)>1 else 'affine',
        out=('fix4_%s.png'%(sys.argv[1] if len(sys.argv)>1 else 'affine')))
