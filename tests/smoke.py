"""CPU integration test with synthetic images and a small ViT; no research data."""
import sys,json,tempfile,importlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
import pandas as pd
import numpy as np
from PIL import Image
from timm.models.vision_transformer import VisionTransformer
import models_vit

torch.set_num_threads(2)
def tiny(**kw):
    kw['global_pool']='avg'
    return VisionTransformer(patch_size=16,embed_dim=32,depth=1,num_heads=4,**kw)
models_vit.RETFound_mae=tiny

def run():
 with tempfile.TemporaryDirectory() as tmp:
    base=Path(tmp); data=base/'data'; labels=json.loads((ROOT/'configs/labels.json').read_text())
    for sub in ['DOVS/train','DOVS/val','DOVS/test','DOVS/external','DOVS/REFUGE2','SMDG/test','REFUGE2/test']:
        for i in range(4):
            d=data/sub/('0_NGON' if i%2==0 else '1_GON'); d.mkdir(parents=True,exist_ok=True)
            Image.fromarray(np.random.default_rng(i).integers(0,256,(224,224,3),dtype=np.uint8)).save(d/f'image_{i}.jpg')
    ld=data/'DOVS/label_multilabel_GON-NGON'; ld.mkdir()
    stems = [('train_GON_RY_20260105','train_NGON_RY_20260211'),
             ('val_GON_RY_20260105','val_NGON_RY20260211'),
             ('test_GON_RY_20260105','test_NGON_RY20260211'),
             ('SMDG_GON_RY_20260105','external_NGON_RY20260211'),
             ('REFUGE2_GON_RY_20260105','REFUGE2_NGON_RY20260211')]
    for gon,ngon in stems:
        for stem,indices in [(gon,[1,3]),(ngon,[0,2])]:
            pd.DataFrame([dict(image=f'image_{i}.jpg',**{label:(i%2 if j<11 else int(i in [0,1])) for j,label in enumerate(labels)}) for i in indices]).to_csv(ld/f'{stem}.csv',index=False)
    checkpoints={}
    for kind in ['binary','multilabel']:
        mod=importlib.import_module('train_binary' if kind=='binary' else 'train_multilabel')
        cfg=json.loads((ROOT/f'configs/{kind}.json').read_text())['training']
        args=mod.get_args_parser().parse_args([]); vars(args).update(cfg)
        vars(args).update(device='cpu',data_path=str(data/'DOVS'),output_dir=str(base/'out'),log_dir=None,num_workers=0,batch_size=4,epochs=1,finetune='',task=kind,train_head_only_epochs=0)
        if kind=='multilabel': args.single_pretrain=str(checkpoints['binary'])
        mod.main(args,torch.nn.CrossEntropyLoss() if kind=='binary' else torch.nn.BCEWithLogitsLoss())
        checkpoints[kind]=base/'out'/kind/'checkpoint-best_0.pth'
        assert checkpoints[kind].exists()
        mod=importlib.import_module('train_multilabel')
        for tta in ['tent']:
            args=mod.get_args_parser().parse_args([])
            vars(args).update(cfg)
            vars(args).update(json.loads((ROOT/f'configs/{kind}.json').read_text())['evaluation'])
            vars(args).update(device='cpu',data_path=str(data/'DOVS'),output_dir=str(base/'out'),log_dir=None,num_workers=0,batch_size=4,finetune='',single_pretrain='',task=kind+('_tent' if tta else '_plain'),eval=True,resume=str(checkpoints[kind]),tta=tta)
            try: mod.main(args,torch.nn.CrossEntropyLoss() if kind=='binary' else torch.nn.BCEWithLogitsLoss())
            except SystemExit as e:
                assert e.code==0
            for split in ['val','test','external','REFUGE2']:
                f=base/'out'/args.task/f'image_predictions_{split}_epoch000.csv'
                df=pd.read_csv(f); assert len(df)==4 and np.isfinite(df.filter(like='_prob').values).all()
                if kind=='multilabel': assert list(df.columns[1:])==[s+'_prob' for s in labels]
    print('PASS: binary/multilabel training, checkpoint saving, final TENT evaluation on all four splits')
if __name__=='__main__': run()
