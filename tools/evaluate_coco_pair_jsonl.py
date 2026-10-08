#!/usr/bin/env python3
"""Paired COCO bbox AP for PT and RKNN JSONL predictions on fixed image IDs."""
import argparse,csv,json,math
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

p=argparse.ArgumentParser();p.add_argument('--annotations',required=True,type=Path)
p.add_argument('--test-list',required=True,type=Path);p.add_argument('--pt-dir',required=True,type=Path)
p.add_argument('--rknn-dir',required=True,type=Path);p.add_argument('--models',required=True)
p.add_argument('--output',required=True,type=Path);a=p.parse_args()
coco=COCO(str(a.annotations));cat_ids=sorted(coco.getCatIds())
ids=[int(Path(x).stem) for x in a.test_list.read_text().splitlines() if x.strip()]
expected=set(ids);rows=[]
for model in [x.strip() for x in a.models.split(',') if x.strip()]:
    values={'model':model,'images':len(ids)}
    for source,root in [('pt',a.pt_dir),('rknn',a.rknn_dir)]:
        raw=[json.loads(x) for x in (root/f'{model}_map.jsonl').read_text().splitlines() if x.strip()]
        seen={int(x['image_id']) for x in raw}
        invalid=0;dets=[]
        for d in raw:
            b=d['bbox']
            if (d['image_id'] not in expected or d['class_id']<0 or d['class_id']>=len(cat_ids) or
                not all(math.isfinite(float(x)) for x in b+[d['score']]) or b[2]<=0 or b[3]<=0):
                invalid+=1;continue
            dets.append({'image_id':d['image_id'],'category_id':cat_ids[d['class_id']],
                         'score':d['score'],'bbox':b})
        dt=coco.loadRes(dets);ev=COCOeval(coco,dt,'bbox');ev.params.imgIds=ids
        ev.evaluate();ev.accumulate();ev.summarize()
        values.update({f'{source}_predictions':len(dets),f'{source}_invalid':invalid,
                       f'{source}_image_ids':len(seen),f'{source}_missing_image_ids':len(expected-seen),
                       f'{source}_map50_95':ev.stats[0],f'{source}_map50':ev.stats[1],
                       f'{source}_map75':ev.stats[2],f'{source}_mar100':ev.stats[8]})
    values['int8_delta_map50_95']=values['rknn_map50_95']-values['pt_map50_95']
    values['int8_delta_map50']=values['rknn_map50']-values['pt_map50']
    rows.append(values)
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
