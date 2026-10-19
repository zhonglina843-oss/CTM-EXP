#!/usr/bin/env python3
"""Train one normalized S projection shared across all CTM ticks."""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score, roc_auc_score


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--trace-dir",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--device",default="cuda:0"); p.add_argument("--epochs",type=int,default=300)
    p.add_argument("--out-dim",type=int,default=32); p.add_argument("--edge-k",type=int,default=8192)
    p.add_argument("--batch-size",type=int,default=32); a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    device=torch.device(a.device if torch.cuda.is_available() else "cpu")
    data=np.load(a.trace_dir/"traces.npz"); post=data["post_state"].astype(np.float32)
    with (a.trace_dir/"manifest.csv").open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    labels=np.asarray([int(r["label"]) for r in rows]); splits=np.asarray([r["split"] for r in rows]); known=splits=="known_train"; novel=splits=="novel_test"
    pairs=np.triu_indices(post.shape[-1],1); pos=np.linspace(0,len(pairs[0])-1,a.edge_k,dtype=np.int64); fi,fj=pairs[0][pos],pairs[1][pos]
    # Build one fixed-coordinate S embedding for every tick; no full S is stored.
    features=np.empty((len(post),post.shape[1],a.edge_k),dtype=np.float32); start=time.time()
    for batch_start in range(0,len(post),a.batch_size):
        batch_end=min(len(post),batch_start+a.batch_size); cumulative=np.zeros((batch_end-batch_start,post.shape[-1],post.shape[-1]),dtype=np.float32)
        for tick in range(post.shape[1]):
            x_tick=post[batch_start:batch_end,tick]; cumulative+=np.einsum("bi,bj->bij",x_tick,x_tick,optimize=True); matrix=cumulative/(tick+1); matrix=(matrix+matrix.swapaxes(1,2))*.5; vector=np.asarray(matrix[:,fi,fj],dtype=np.float32); vector/=np.maximum(np.linalg.norm(vector,axis=1,keepdims=True),1e-8); features[batch_start:batch_end,tick]=vector
        print(f"[DYNAMIC-NSCL] built batch={batch_end}/{len(post)} elapsed={time.time()-start:.1f}s",flush=True)
    x=torch.tensor(features,device=device)
    classes=sorted(set(labels[known].tolist())); train_novel=np.zeros(len(rows),bool); train_novel[np.flatnonzero(novel)[::2]]=True; train_mask=known|train_novel
    train_indices=np.flatnonzero(train_mask).tolist(); pos_pairs=[]; neg_pairs=[]
    for i in train_indices:
        for j in train_indices:
            if i<j: (pos_pairs if known[i] and known[j] and labels[i]==labels[j] else neg_pairs).append((i,j))
    projection=nn.Sequential(nn.Linear(a.edge_k,256),nn.GELU(),nn.Linear(256,a.out_dim)).to(device); opt=torch.optim.AdamW(projection.parameters(),lr=2e-3)
    for epoch in range(a.epochs):
        losses=[]
        for tick in range(x.shape[1]):
            z=nn.functional.normalize(projection(x[:,tick]),dim=1); gram=z@z.T
            loss=(1-torch.stack([gram[i,j] for i,j in pos_pairs]).mean()) + (torch.stack([gram[i,j]**2 for i,j in neg_pairs]).mean())
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.item()))
        if epoch%25==0 or epoch==a.epochs-1: print(f"[DYNAMIC-NSCL] epoch={epoch+1}/{a.epochs} loss={np.mean(losses):.6f}",flush=True)
    with torch.no_grad():
        embeddings=[]
        for tick in range(x.shape[1]): embeddings.append(nn.functional.normalize(projection(x[:,tick]),dim=1).cpu().numpy())
    z=np.asarray(embeddings).transpose(1,0,2); train_z=z[known]; train_y=labels[known]; proto=np.stack([train_z[train_y==c].mean(axis=(0,1)) for c in classes]); proto/=np.maximum(np.linalg.norm(proto,axis=1,keepdims=True),1e-8)
    novelty=1-(z@proto.T).max(axis=2); eval_mask=(splits=="known_test")|(splits=="novel_test"); y=(splits[eval_mask]=="novel_test").astype(int)
    scores={"dynamic_nscl_mean":novelty.mean(1),"dynamic_nscl_late":novelty[:,novelty.shape[1]//2:].mean(1),"dynamic_nscl_final":novelty[:,-1]}; metrics={}
    for name,score in scores.items(): metrics[name]={"auroc":float(roc_auc_score(y,score[eval_mask])),"aupr_novel":float(average_precision_score(y,score[eval_mask]))}
    np.savez_compressed(a.output_dir/"dynamic_nscl_projection.npz",embedding=z,novelty=novelty)
    (a.output_dir/"dynamic_nscl_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(json.dumps(metrics,indent=2),flush=True)

if __name__=="__main__": main()
