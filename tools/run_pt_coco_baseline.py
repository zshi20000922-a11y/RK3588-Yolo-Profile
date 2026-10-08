#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from ultralytics import YOLO
def main():
 p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--list',required=True);p.add_argument('--output',required=True);p.add_argument('--models',default='yolov8n,yolov8s,yolo11n,yolo11s,yolo26n,yolo26s');p.add_argument('--device',default='0');a=p.parse_args()
 paths=[x.strip() for x in Path(a.list).read_text().splitlines() if x.strip()];out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 for name in [x.strip() for x in a.models.split(',') if x.strip()]:
  print(name,flush=True); model=YOLO(str(Path(a.weights)/f'{name}.pt'))
  with (out/f'{name}_pt.jsonl').open('w') as f:
   for i,r in enumerate(model.predict(paths,imgsz=640,conf=.001,iou=.45,max_det=100,device=a.device,verbose=False,stream=True)):
    image_id=int(Path(paths[i]).stem)
    for box,score,cls in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy(),r.boxes.cls.cpu().numpy()):
     x1,y1,x2,y2=map(float,box);f.write(json.dumps({'image_id':image_id,'class_id':int(cls),'score':float(score),'bbox':[x1,y1,x2-x1,y2-y1]})+'\n')
    if (i+1)%100==0:print(name,i+1,flush=True)
if __name__=='__main__':main()
