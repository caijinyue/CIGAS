"""Check a trusted BagViT checkpoint without training or reading research images."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('checkpoint',type=Path)
p.add_argument('--classes',type=int,choices=[2,12],required=True)
p.add_argument('--attention',choices=['window','local-mask'],default='window')
a=p.parse_args()
import torch
import models_vit
torch.set_num_threads(4)
c=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True)
m=models_vit.RETFound_mae(img_size=224,num_classes=a.classes,drop_path_rate=.2,global_pool=True)
fn=models_vit.replace_attn_with_win_attn if a.attention=='window' else models_vit.replace_attn_with_local_mask
fn(m,win_size=7,cls_global=False)
m.load_state_dict(c['model'],strict=True); m.eval()
with torch.no_grad(): y=m(torch.zeros(1,3,224,224))
assert y.shape==(1,a.classes) and torch.isfinite(y).all()
print(json.dumps({'strict_load':True,'epoch':c.get('epoch'),'output_shape':list(y.shape),'finite':True}))
