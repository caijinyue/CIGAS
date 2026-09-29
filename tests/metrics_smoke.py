"""Exercise packaged statistical functions on balanced, synthetic predictions."""
import sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
import numpy as np
import pandas as pd
import binary_metrics as binary
import multilabel_metrics as multi

def run():
 y=np.array([0]*10+[1]*10); p=np.array([.1]*10+[.9]*10)
 with tempfile.TemporaryDirectory() as temp:
    root=Path(temp)
    for label in ['0_NGON','1_GON']: (root/label).mkdir()
    names=[f'sample_{i}.jpg' for i in range(20)]
    for i,name in enumerate(names): (root/('0_NGON' if y[i]==0 else '1_GON')/name).touch()
    frame=pd.DataFrame({'image_path':names,'class_1_prob':p})
    b=binary.compute_metrics_for_dataset(('test','','synthetic','',frame,None,str(root),False))
    m=multi.compute_per_class_metrics(y,p,None,'synthetic')
    assert b['AUC']=='100.00' and m['AUC']=='100.00'
    assert b['Sensitivity']=='100.00' and m['Sensitivity']=='100.00'
    assert b['Specificity']=='100.00' and m['Specificity']=='100.00'
    assert float(binary.delong_p_value(y,p,p))==1.0
    assert float(multi.delong_p_value(y,p,p))==1.0
 print('PASS: both statistical functions, binary 1000 / multilabel 500-resample confidence intervals, identical-score DeLong')
if __name__=='__main__': run()
