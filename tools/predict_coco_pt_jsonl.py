#!/usr/bin/env python3
"""Run one Ultralytics PT checkpoint on a fixed COCO image list and write JSONL."""
import argparse, json
from pathlib import Path
from ultralytics import YOLO

p=argparse.ArgumentParser()
p.add_argument('--weights',required=True,type=Path)
p.add_argument('--images',required=True,type=Path,help='one absolute image path per line')
p.add_argument('--output',required=True,type=Path)
p.add_argument('--imgsz',type=int,default=640)
p.add_argument('--conf',type=float,default=0.001)
p.add_argument('--iou',type=float,default=0.45)
p.add_argument('--max-det',type=int,default=100)
p.add_argument('--device',default='cpu')
a=p.parse_args()
model=YOLO(str(a.weights))
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('w') as f:
    for i,line in enumerate(x.strip() for x in a.images.read_text().splitlines() if x.strip()):
        path=Path(line)
        result=model.predict(str(path),imgsz=a.imgsz,conf=a.conf,iou=a.iou,max_det=a.max_det,
                             device=a.device,verbose=False)[0]
        image_id=int(path.stem)
        if result.boxes is not None:
            for box,score,cls in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),
                                     result.boxes.cls.cpu().tolist()):
                x1,y1,x2,y2=box
                f.write(json.dumps({'image_id':image_id,'class_id':int(cls),'score':score,
                                    'bbox':[x1,y1,x2-x1,y2-y1]},separators=(',',':'))+'\n')
        if (i+1)%100==0: print(f'{i+1}/{len([x for x in a.images.read_text().splitlines() if x.strip()])}',flush=True)
