#!/usr/bin/env python3
"""Train an NSCL projection head on frozen S embeddings.

This is the actual Eq. (4)-style sample-graph experiment. CTM is frozen; the
projection head is the only trainable component. Novel samples are used in a
transductive NCD split, with half held out for evaluation.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import torch
from torch import nn


def main():
    p=argparse.ArgumentParser(); p.add_argument("--trace-dir",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--device",default="cuda:0"); p.add_argument("--epochs",type=int,default=300); p.add_argument("--out-dim",type=int,default=32); p.add_argument("--alpha",type=float,default=1.0); p.add_argument("--beta",type=float,default=1.0); p.add_argument("--edge-k",type=int,default=8192); a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    device=torch.device(a.device if torch.cuda.is_available() else "cpu")
    data=np.load(a.trace_dir/"traces.npz"); post=data["post_state"].astype(np.float32)
    with (a.trace_dir/"manifest.csv").open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    labels=np.array([int(r["label"]) for r in rows]); splits=np.array([r["split"] for r in rows]); known=splits=="known_train"; novel=splits=="novel_test"
    # One final S embedding per sample: keep the strongest positive upper-triangle edges.
    x=[]
    for sample in post:
        s=np.einsum("th,tk->hk",sample,sample)/sample.shape[0]; s=(s+s.T)*.5; ii,jj=np.triu_indices(s.shape[0],1); w=np.maximum(s[ii,jj],0); keep=np.argpartition(w,-min(a.edge_k,len(w)))[-min(a.edge_k,len(w)):]; vector=np.zeros(a.edge_k,np.float32); values=w[keep]; vector[:len(values)]=values/(np.linalg.norm(values)+1e-8); x.append(vector)
    x=torch.tensor(np.asarray(x),device=device); known_idx=torch.tensor(known,device=device); train_novel=np.zeros(len(rows),bool); novel_positions=np.flatnonzero(novel); train_novel[novel_positions[::2]]=True; train_mask=torch.tensor(known|train_novel,device=device)
    labels_t=torch.tensor(labels,device=device); classes=sorted(set(labels[known])); known_labels=torch.tensor(classes,device=device)
    projection=nn.Sequential(nn.Linear(a.edge_k,256),nn.GELU(),nn.Linear(256,a.out_dim)).to(device); opt=torch.optim.AdamW(projection.parameters(),lr=2e-3)
    # Graph weights: same known class pairs and same-sample temporal views are represented here by fixed positive pairs.
    pos=[]; neg=[]
    train_indices=torch.nonzero(train_mask).flatten().tolist()
    for i in train_indices:
        for j in train_indices:
            if i<j:
                (pos if (known[i] and known[j] and labels[i]==labels[j]) else neg).append((i,j))
    for epoch in range(a.epochs):
        z=projection(x); zi=z[:,None,:]; zj=z[None,:,:]; gram=torch.matmul(z, z.T); pos_loss=-2*a.alpha*torch.stack([gram[i,j] for i,j in pos]).mean(); neg_loss=(torch.stack([gram[i,j]**2 for i,j in neg]).mean() if neg else torch.tensor(0.,device=device)); loss=pos_loss+neg_loss; opt.zero_grad(); loss.backward(); opt.step()
        if epoch%25==0 or epoch==a.epochs-1: print(f"[NSCL] epoch={epoch+1}/{a.epochs} loss={loss.item():.6f} pos={len(pos)} neg={len(neg)}",flush=True)
    with torch.no_grad():
        z=projection(x).cpu().numpy(); train_z=z[known]; train_y=labels[known]; proto=np.stack([train_z[train_y==c].mean(0) for c in classes]); proto/=np.maximum(np.linalg.norm(proto,axis=1,keepdims=True),1e-8); novelty=1-(z@proto.T).max(1)
    np.savez_compressed(a.output_dir/"nscl_projection.npz",embedding=z,novelty=novelty)
    (a.output_dir/"nscl_manifest.json").write_text(json.dumps({"embedding":"S_T top-positive-edge vector","edge_k":a.edge_k,"out_dim":a.out_dim,"epochs":a.epochs,"ctm_frozen":True,"transductive_novel_train":True},indent=2),encoding="utf-8")
    print(f"[NSCL] wrote {a.output_dir / 'nscl_projection.npz'}", flush=True)

if __name__=="__main__": main()
