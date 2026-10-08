#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
import cv2
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--list',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output).resolve();frames=out/'frames';frames.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for index,line in enumerate(Path(a.list).read_text().splitlines()):
        src=Path(line.strip()); image=cv2.imread(str(src))
        if image is None: raise RuntimeError(f'cannot decode {src}')
        h,w=image.shape[:2]; eh=(h+1)&~1; ew=(w+15)&~15
        if (eh,ew)!=(h,w): image=cv2.copyMakeBorder(image,0,eh-h,0,ew-w,cv2.BORDER_CONSTANT,value=(114,114,114))
        i420=cv2.cvtColor(image,cv2.COLOR_BGR2YUV_I420).reshape(-1)
        y=i420[:ew*eh]; u=i420[ew*eh:ew*eh+ew*eh//4]; v=i420[ew*eh+ew*eh//4:]
        uv=np.empty(ew*eh//2,dtype=np.uint8);uv[0::2]=u;uv[1::2]=v
        image_id=int(src.stem); target=frames/f'{image_id:012d}.nv12'
        with target.open('wb') as f: f.write(y.tobytes());f.write(uv.tobytes())
        manifest.append((target,ew,eh,image_id,str(src),w,h))
        if (index+1)%100==0: print(f'{index+1}/1000',flush=True)
    (out/'input_list.txt').write_text(''.join(f'{x[0]} {x[1]} {x[2]} {x[3]}\n' for x in manifest))
    meta={'count':len(manifest),'source_list':str(Path(a.list).resolve()),'format':'NV12','padding_policy':'pad right to 16-pixel stride and bottom to even height with 114','items':[{'image_id':x[3],'source':x[4],'source_width':x[5],'source_height':x[6],'nv12_width':x[1],'nv12_height':x[2]} for x in manifest]}
    (out/'manifest.json').write_text(json.dumps(meta,indent=2))
    digest=hashlib.sha256((out/'input_list.txt').read_bytes()).hexdigest();print('manifest_sha256',digest)
if __name__=='__main__':main()
