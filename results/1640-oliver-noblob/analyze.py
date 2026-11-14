import sys, os, json, math, collections
sys.path.insert(0,'/home/oliver/work/ab-r21')
import cv2, numpy as np, pandas as pd
from aaltonomous.baselines.classical.config import load_config
from aaltonomous.baselines.classical.perception import Perception
from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene
TREE='/home/oliver/work/ab-r21'; D='/home/oliver/work/e1-audit/results/1148-opencode-e1'
os.chdir(TREE)
cfg=load_config(overlay='finals_r21'); p=Perception(cfg,Scene.load(),Calibration.load())
fidx=json.load(open(D+'/frame_index.json'))
inv={v:k for k,v in fidx.items()}
df=pd.read_parquet('/var/tmp/hf_drop2/data/chunk-000/file-000.parquet',columns=['index','observation.state'])
states={int(i):np.asarray(s,float) for i,s in zip(df['index'],df['observation.state'])}
L=[json.loads(l) for l in open(D+'/labels_clean.jsonl')]
objs=[r for r in L if r.get('kind')=='object' and r['ep']<40]
byf=collections.defaultdict(list)
for r in objs: byf[r['frame_idx']].append(r)
band={'orange':'red','purple':'blue','yellow':'yellow'}
OUT=[]
MINA=cfg['perception']['min_area_px'] if 'perception' in cfg else 120
for gi,rows in byf.items():
    name=inv.get(gi)
    if name is None: continue
    rgb=cv2.cvtColor(cv2.imread(f'/var/tmp/hf_drop2/frames/{name}.png'),cv2.COLOR_BGR2RGB)
    s=states.get(gi); rq=(s[:14],s[14:28]) if s is not None and len(s)>=28 else None
    dets,rej=p.detect(rgb,None,debug=True,include_bins=False,robot_q=rq)
    allu=[d.uv for d in dets]+[d.uv for d,_ in rej]
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    keep=p._keep_mask(None,exclude_bins=True)
    keep0=keep.copy()
    if rq is not None:
        arm=p._fk_arm_mask(keep.shape,*rq)
        if arm is not None: keep=keep.copy(); keep[arm]=0
    armmask=(keep0>0)&(keep==0)
    for g in rows:
        cx,cy=g['centroid_px']
        near=any(math.hypot(u[0]-cx,u[1]-cy)<=25 for u in allu)
        x0,y0,x1,y1=[int(v) for v in g['bbox_px']]; x0=max(x0,0);y0=max(y0,0)
        sub=hsv[y0:y1+1,x0:x1+1]
        col=g['colour']; rng=p.hsv[band[col]]
        m=np.zeros(hsv.shape[:2],np.uint8)
        for lo,hi in rng: m|=cv2.inRange(hsv,np.array(lo,np.uint8),np.array(hi,np.uint8))
        mo=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        mk=mo&keep; mk=cv2.morphologyEx(mk,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        def ccmax(mm,full=False):
            n,lab,st,ce=cv2.connectedComponentsWithStats(mm,connectivity=8)
            best=(0,None)
            for i in range(1,n):
                bx,by,bw,bh=st[i,:4]
                if bx<=x1 and bx+bw>=x0 and by<=y1 and by+bh>=y0:
                    # component pixels inside bbox
                    inb=int((lab[y0:y1+1,x0:x1+1]==i).sum())
                    if inb>best[0]: best=(inb,(int(st[i,4]),ce[i].tolist()))
            return best
        rawb=int((m[y0:y1+1,x0:x1+1]>0).sum()); area=(y1-y0+1)*(x1-x0+1)
        c_raw=ccmax(m); c_open=ccmax(mo); c_keep=ccmax(mk)
        H,S_,V=sub[...,0].astype(int),sub[...,1].astype(int),sub[...,2].astype(int)
        # object pixels proxy: chromatic pixels (S>=35,V>=60) in bbox
        chrom=(S_>=35)&(V>=60)
        armfrac=float(armmask[y0:y1+1,x0:x1+1].mean())
        hm=g['hsv_median']
        OUT.append(dict(frame=name,ep=g['ep'],col=col,shape=g['shape'],noblob=not near,cls=f"{col}-{g['shape']}",
            bbox=g['bbox_px'],bbox_area=area,raw_in_bbox=rawb,raw_frac=rawb/area,
            raw_cc=c_raw[0],raw_cc_area=(c_raw[1] or [0])[0],open_cc=c_open[0],keep_cc=c_keep[0],keep_cc_area=(c_keep[1] or [0])[0],
            keep_cc_cent=(c_keep[1] or [0,None])[1],gt_cent=[cx,cy],armfrac=armfrac,hsv_med=hm,note=g.get('note'),
            chrom_frac=float(chrom.mean()),
            H_ok=None))
json.dump(OUT,open(sys.argv[1],'w'))
print(len(OUT),sum(o['noblob'] for o in OUT))
