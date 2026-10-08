#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
def main():
 p=argparse.ArgumentParser();p.add_argument('--annotations',required=True);p.add_argument('--test-list',required=True);p.add_argument('--results',required=True);p.add_argument('--output',required=True);p.add_argument('--models',default='yolov8n,yolov8s,yolo11n,yolo11s,yolo26n,yolo26s');a=p.parse_args()
 coco=COCO(a.annotations);cats=sorted(coco.getCatIds());ids=[int(Path(x).stem) for x in Path(a.test_list).read_text().splitlines() if x.strip()];rows=[]
 for m in [x.strip() for x in a.models.split(',') if x.strip()]:
  raw=[json.loads(x) for x in (Path(a.results)/f'{m}_pt.jsonl').read_text().splitlines() if x.strip()]
  det=[{'image_id':d['image_id'],'category_id':cats[d['class_id']],'score':d['score'],'bbox':d['bbox']} for d in raw]
  ev=COCOeval(coco,coco.loadRes(det),'bbox');ev.params.imgIds=ids;ev.evaluate();ev.accumulate();ev.summarize()
  rows.append({'model':m,'images':len(ids),'predictions':len(det),'map50_95':ev.stats[0],'map50':ev.stats[1],'map75':ev.stats[2],'mar100':ev.stats[8]})
 with Path(a.output).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
