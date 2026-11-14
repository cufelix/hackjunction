import json,collections,numpy as np
S='/tmp/claude-1000/-home-oliver-hackjunction/8d52d24e-7c5d-41dd-81c4-7a0e7f626189/scratchpad/nb'
rows=json.load(open(S+'/rows.json')); pg=json.load(open(S+'/base/fair_recall.json'))['per_gt']
# per_gt is in label order same as objs order -> join by (frame,centroid)
key={(r['frame'],tuple(r['gt_cent'])):r for r in rows}
miss=[];vis=0
for g in pg:
    if g['locus']!='visible_on_mat': continue
    vis+=1
    if g['matched']: continue
    r=key[(g['frame'],tuple(g['centroid']))]; r['cls']=g['cls']; miss.append(r)
print('visible',vis,'missed',len(miss),'noblob',sum(r['noblob'] for r in miss))
BAND={'orange':'H<=22|H>=170,S>=70,V>=70','purple':'H100-145,S>=70,V>=50','yellow':'H24-60,S>=35,V>=90'}
def cause(r):
    h,s,v=r['hsv_med']; c=r['col']
    A=r['bbox_area']
    if r['keep_cc']>=120 and r['keep_cc']>=0 and r['keep_cc_area']>=120: return 'merged/offset (>=120px blob overlaps, centroid >25px away)'
    if r['raw_in_bbox']<60 or r['raw_cc']<60:
        # no meaningful mask pixels
        if c=='orange': hue_ok = h<=22 or h>=170
        elif c=='purple': hue_ok=100<=h<=145
        else: hue_ok=24<=h<=60
        if not hue_ok: return 'hue out of band'
        fl={'orange':(70,70),'purple':(70,50),'yellow':(35,90)}[c]
        if s<fl[0]: return 'S below floor'
        if v<fl[1]: return 'V below floor'
        return 'median in band, mask sparse (other)'
    if r['open_cc']<120 and r['raw_cc']>=120: return 'opening 3x3 kills blob'
    if r['open_cc']>=120 and r['keep_cc']<120:
        return 'FK arm mask / keep mask removes' if r['armfrac']>0.05 else 'keep mask removes (bin/depth)'
    if r['raw_cc']<120: return 'fragmented / area below 120 floor (raw CC<120)'
    return 'other'
out=collections.defaultdict(collections.Counter); meds=collections.defaultdict(lambda: collections.defaultdict(list))
for r in miss:
    if not r['noblob']: continue
    k=cause(r); out[r['cls']][k]+=1; meds[r['cls']][k].append(r['hsv_med'])
    r['cause']=k
causes=sorted({k for c in out.values() for k in c})
print('| class | n | '+' | '.join(causes)+' |')
for cl,c in sorted(out.items()):
    print('|',cl,'|',sum(c.values()),'|',' | '.join(str(c.get(k,0)) for k in causes),'|')
print()
for cl in sorted(meds):
    for k,v in meds[cl].items():
        a=np.array(v); print(cl,k,len(v),'median HSV',np.median(a,0).tolist(),'H p10/p90',np.percentile(a[:,0],[10,90]).tolist(),'S p10/p90',np.percentile(a[:,1],[10,90]).tolist(),'V p10/p90',np.percentile(a[:,2],[10,90]).tolist())
# also area stats
import statistics
for cl in sorted(out):
    ar=[r['bbox_area'] for r in miss if r['noblob'] and r['cls']==cl]; print(cl,'bbox area med',statistics.median(ar),'raw_frac med',statistics.median([r['raw_frac'] for r in miss if r['noblob'] and r['cls']==cl]))
json.dump([r for r in miss if r['noblob']],open(S+'/noblob.json','w'))
