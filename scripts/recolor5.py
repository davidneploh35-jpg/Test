import numpy as np, json
from PIL import Image
from hairmask import hair_mask

SRC='/root/.claude/uploads/92ed6dba-dbfa-5416-bab6-95a063ee99a3/78db79cd-image.jpg'

def run(out_png, out_jpg):
    st=json.load(open('hairstats.json'))
    a=np.asarray(Image.open(SRC).convert('RGB')).astype(np.float64)
    lt=np.load('lm_i5.npy')
    m,_,_=hair_mask(a,lt)

    tgt_mean=np.mean([st[k]['mean'] for k in ('photo2','photo3','photo4')],axis=0)
    tgt_std =np.mean([st[k]['std']  for k in ('photo2','photo3','photo4')],axis=0)
    tgt_bg  =np.mean([st[k]['bg']   for k in ('photo2','photo3','photo4')])
    exp = st['photo5']['bg']/tgt_bg                      # this frame is lit slightly lower
    tgt_mean=np.array(tgt_mean)*exp; tgt_std=np.array(tgt_std)*exp
    src_mean=np.array(st['photo5']['mean']); src_std=np.array(st['photo5']['std'])

    gain=np.clip(tgt_std/src_std, 0.9, 1.1)              # keep this frame's own hair contrast
    off =tgt_mean-gain*src_mean
    print('exposure factor %.3f'%exp)
    print('gain',np.round(gain,3),'offset',np.round(off,1))
    print('src mean',np.round(src_mean,1),'-> target',np.round(tgt_mean,1))

    new=a*gain+off
    out=a*(1-m[:,:,None])+new*m[:,:,None]
    out=np.clip(np.rint(out),0,255).astype(np.uint8)
    Image.fromarray(out).save(out_png)
    Image.fromarray(out).save(out_jpg, quality=97, subsampling=0)

    chk=out.astype(float)[m>0.7]
    print('result hair mean',np.round(chk.mean(axis=0),1),'std',np.round(chk.std(axis=0),1))
    iod=np.linalg.norm(lt[33]-lt[263])
    box=(int(lt[:,0].min()-1.5*iod),0,int(lt[:,0].max()+1.5*iod),int(lt[152,1]+0.6*iod))
    Image.fromarray(out).crop(box).save('r5_new.png')
    Image.fromarray(a.astype(np.uint8)).crop(box).save('r5_old.png')

if __name__=='__main__':
    run('/home/user/Test/photos/photo_5_haircolor.png','/home/user/Test/photos/photo_5_haircolor.jpg')
