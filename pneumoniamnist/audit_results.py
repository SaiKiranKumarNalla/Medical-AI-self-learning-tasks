"""Re-evaluate saved learning checkpoints and generate traceable error examples.
No training or model selection occurs here. Test results remain exploratory.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix
from train_replication import ResNet18
from run_residual_ablation import PlainNet18

ROOT = Path(__file__).resolve().parent

def main():
    torch.set_num_threads(4)
    data = np.load(ROOT / 'data/pneumoniamnist.npz')
    x = (torch.tensor(data['test_images'][:, None], dtype=torch.float32) / 255 - .5) / .5
    y = data['test_labels'].ravel()
    results = {'device': 'cpu', 'status': 'post-hoc checkpoint evaluation, not retraining', 'models': {}}
    sets = {s: {hashlib.sha256(a.tobytes()).hexdigest() for a in data[s+'_images']} for s in ['train','val','test']}
    results['duplicate_image_hashes'] = {a+'_'+b: len(sets[a]&sets[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    models = [('baseline', 'best_resnet18_val_auc.pth', ResNet18), ('weighted', 'best_resnet18_weighted.pth', ResNet18), ('focal', 'best_resnet18_focal.pth', ResNet18), ('plain', 'best_plainnet18.pth', PlainNet18)]
    out = ROOT / 'results';out.mkdir(exist_ok=True)
    for name, filename, cls in models:
        model = cls()
        path = ROOT/'checkpoints'/filename
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));model.eval()
        with torch.inference_mode():
            probs = torch.cat([model(t).softmax(1)[:,1] for t in x.split(64)]).numpy()
        preds = (probs>=.5).astype(int)
        tn,fp,fn,tp = confusion_matrix(y,preds,labels=[0,1]).ravel()
        results['models'][name] = dict(auc=float(roc_auc_score(y,probs)),accuracy=float((tn+tp)/len(y)),sensitivity=float(tp/(tp+fn)),specificity=float(tn/(tn+fp)),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),parameters=sum(p.numel() for p in model.parameters()),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        np.savetxt(out/f'{name}_predictions.csv',np.column_stack([np.arange(len(y)),y,probs,preds]),delimiter=',',header='test_index,target,p_pneumonia,prediction',comments='')
        if name=='baseline':
            # Choose examples by measured labels/predictions, never hard-code categories.
            fig,axes=plt.subplots(4,3,figsize=(9,11))
            examples=[]
            for row,(category,target,pred) in enumerate([('TP',1,1),('TN',0,0),('FP',0,1),('FN',1,0)]):
                idx=int(np.flatnonzero((y==target)&(preds==pred))[0])
                features=[]
                handle=model.layer4.register_forward_hook(lambda module,inputs,output: features.append(output))
                logits=model(x[idx:idx+1]);activation=features[0]
                # Attribute the pneumonia logit consistently across all categories.
                gradient=torch.autograd.grad(logits[0,1],activation)[0]
                cam=F.relu((gradient.mean((2,3),keepdim=True)*activation).sum(1,keepdim=True))
                cam=F.interpolate(cam,size=(28,28),mode='bilinear',align_corners=False)[0,0].detach().numpy()
                peak=float(cam.max());cam=cam/peak if peak>0 else cam
                handle.remove()
                image=data['test_images'][idx]
                axes[row,0].imshow(image,cmap='gray');axes[row,0].set_title(f'{category}: test {idx}; y={target}, pred={pred}\np(pneumonia)={probs[idx]:.4f}')
                axes[row,1].imshow(cam,cmap='jet',vmin=0,vmax=1);axes[row,1].set_title('Grad-CAM: pneumonia'+ (' (zero)' if peak==0 else ''))
                axes[row,2].imshow(image,cmap='gray');axes[row,2].imshow(cam,cmap='jet',vmin=0,vmax=1,alpha=.35);axes[row,2].set_title('Overlay (not a lesion mask)' )
                for ax in axes[row]:ax.axis('off')
                examples.append(dict(category=category,test_index=idx,target=target,prediction=pred,cam_max_before_normalization=peak))
            fig.tight_layout();fig.savefig(ROOT/'figures/gradcam_error_analysis.png',dpi=140);plt.close(fig)
            results['gradcam_examples']=examples
    (out/'verified_metrics.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))

if __name__=='__main__': main()
