#!/usr/bin/env python3
import argparse,csv,json,math,statistics
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def pct(v,q):
    v=sorted(v);p=(len(v)-1)*q;i=int(p);j=min(i+1,len(v)-1);return v[i]*(j-p)+v[j]*(p-i)
def main():
    p=argparse.ArgumentParser();p.add_argument('--annotations',required=True);p.add_argument('--test-list',required=True);p.add_argument('--results',required=True);p.add_argument('--output',required=True);p.add_argument('--models',default='yolov8n,yolov8s,yolo11n,yolo11s,yolo26n,yolo26s');a=p.parse_args()
    coco=COCO(a.annotations);cat_ids=sorted(coco.getCatIds());ids=[int(Path(x).stem) for x in Path(a.test_list).read_text().splitlines() if x.strip()]
    root=Path(a.results);rows=[]
    for model in [x.strip() for x in a.models.split(',') if x.strip()]:
        raw=[json.loads(x) for x in (root/f'{model}_map.jsonl').read_text().splitlines() if x.strip()]
        detections=[];invalid=0
        for d in raw:
            box=d['bbox']
            if d['class_id']<0 or d['class_id']>=len(cat_ids) or not all(math.isfinite(float(x)) for x in box+[d['score']]) or box[2]<=0 or box[3]<=0: invalid+=1;continue
            detections.append({'image_id':d['image_id'],'category_id':cat_ids[d['class_id']],'score':d['score'],'bbox':box})
        dt=coco.loadRes(detections);ev=COCOeval(coco,dt,'bbox');ev.params.imgIds=ids;ev.evaluate();ev.accumulate();ev.summarize()
        timing=list(csv.DictReader((root/f'{model}_timings.csv').open())); stages={}
        for k in ['rga_ms','inference_ms','output_sync_ms','postprocess_ms','e2e_ms']:
            v=[float(x[k]) for x in timing];stages[k+'_mean']=statistics.fmean(v);stages[k+'_p50']=pct(v,.5);stages[k+'_p95']=pct(v,.95);stages[k+'_p99']=pct(v,.99)
        rows.append({'model':model,'images':len(ids),'predictions':len(detections),'invalid':invalid,'map50_95':ev.stats[0],'map50':ev.stats[1],'map75':ev.stats[2],'mar100':ev.stats[8],**stages})
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
