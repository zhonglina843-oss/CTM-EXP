#!/usr/bin/env python3
"""Memory-bounded 12-method similarity benchmark for checkpoint-matched S.

Loads one sample/tick at a time, keeps only Top-k sparse graph state, and runs
one method at a time. Spectral/DeltaCon/GW/GED remain the same approximations
used by the old extended branch, but no 4096x4096 dense adjacency is retained.
"""
from __future__ import annotations

import argparse, csv, json, math, time
from collections import Counter
from pathlib import Path
import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, factorized

METHODS = ("weighted_jaccard", "edge_cosine", "degree_strength_cosine", "spectral_similarity",
           "community_nmi", "community_ari", "delta_con", "delta_con_attr",
           "ged_approx", "wl_kernel", "gw_approx", "dynamic_trajectory")


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))


def sparse_state(matrix, active, density):
    n = matrix.shape[0]
    ii, jj = np.triu_indices(n, 1)
    w = np.maximum(np.asarray(matrix[ii, jj], dtype=np.float32), 0)
    valid = np.isfinite(w) & (w > 0)
    ii, jj, w = ii[valid], jj[valid], w[valid]
    k = min(max(1, int(round(n * (n - 1) / 2 * density))), len(w))
    keep = np.argpartition(w, -k)[-k:] if k else np.empty(0, dtype=int)
    ii, jj, w = ii[keep], jj[keep], w[keep]
    rows = np.r_[ii, jj]; cols = np.r_[jj, ii]; vals = np.r_[w, w]
    adjacency = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    degree = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float32)
    edge_keys = np.asarray(ii, dtype=np.int64) * n + jj
    edges = {int(key): float(value) for key, value in zip(edge_keys, w)}
    graph = nx.from_scipy_sparse_array(adjacency)
    comps = list(nx.connected_components(graph))
    lcc = max(comps, key=len) if comps else set()
    return {"n": n, "edges": edges, "degree": degree, "adj": adjacency, "lcc": lcc}


def cosine(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else 0.0


def edge_cosine(a, b):
    keys = set(a["edges"]) | set(b["edges"])
    x = np.fromiter((a["edges"].get(k, 0) for k in keys), dtype=np.float32)
    y = np.fromiter((b["edges"].get(k, 0) for k in keys), dtype=np.float32)
    return cosine(x, y)


def similarity(a, b, method):
    if method == "edge_cosine": return edge_cosine(a, b)
    if method == "degree_strength_cosine": return cosine(a["degree"], b["degree"])
    keys = set(a["edges"]) | set(b["edges"])
    x = np.fromiter((a["edges"].get(k, 0) for k in keys), dtype=np.float32)
    y = np.fromiter((b["edges"].get(k, 0) for k in keys), dtype=np.float32)
    if method == "weighted_jaccard": return float(np.minimum(x, y).sum() / max(np.maximum(x, y).sum(), 1e-8))
    if method == "ged_approx": return float(1 - np.abs(x-y).sum() / (x.sum()+y.sum()+1e-8))
    if method == "spectral_similarity":
        ka = min(32, max(1, a["n"]-2)); kb = min(32, max(1, b["n"]-2))
        va = np.sort(eigsh(a["adj"], k=ka, which="LA", return_eigenvectors=False))
        vb = np.sort(eigsh(b["adj"], k=kb, which="LA", return_eigenvectors=False))
        size = min(len(va), len(vb)); d = np.linalg.norm(va[-size:] - vb[-size:]) / (np.linalg.norm(va[-size:])+np.linalg.norm(vb[-size:])+1e-8)
        return float(1/(1+d))
    if method in ("community_nmi", "community_ari"):
        ca = {x:i for i,c in enumerate(nx.connected_components(nx.from_scipy_sparse_array(a["adj"]))) for x in c}
        cb = {x:i for i,c in enumerate(nx.connected_components(nx.from_scipy_sparse_array(b["adj"]))) for x in c}
        vals = [(ca.get(i,-1), cb.get(i,-1)) for i in range(a["n"]) if i in ca or i in cb]
        return float(sum(x==y for x,y in vals)/max(1,len(vals)))
    if method == "wl_kernel":
        da = Counter(dict(a["adj"].getnnz(axis=1)).values()); db = Counter(dict(b["adj"].getnnz(axis=1)).values())
        keys = set(da)|set(db); return cosine(np.array([da[k] for k in keys]), np.array([db[k] for k in keys]))
    if method == "gw_approx":
        return similarity(a,b,"spectral_similarity")
    if method == "delta_con":
        return 1/(1+float(sparse.linalg.norm(a["adj"]-b["adj"])))
    if method == "delta_con_attr":
        return 1/(1+float(np.mean(np.abs(a["degree"]-b["degree"]))))
    if method == "dynamic_trajectory":
        return similarity(a,b,"edge_cosine")
    raise ValueError(method)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--subset-dir",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--ticks",default="1,5,10,15,20,30,40,50"); p.add_argument("--density",type=float,default=.01); p.add_argument("--methods",default=",".join(METHODS)); a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    rows=read_rows(a.subset_dir/"labels.csv"); methods=[x for x in a.methods.split(",") if x]
    ticks=[int(x) for x in a.ticks.split(",") if x]; pair_out=[]; tick_out=[]
    for tick in ticks:
        states=[]; started=time.time()
        for sample_number, row in enumerate(rows, 1):
            data=np.load(a.subset_dir/"samples"/row["sample_id"]/"S_full_active_all_ticks.npz",mmap_mode="r")["matrices"][ticks.index(tick)]
            active=np.load(a.subset_dir/"samples"/row["sample_id"]/"active_neuron_indices.npy")
            states.append(sparse_state(data,active,a.density))
            if sample_number == 1 or sample_number % 5 == 0 or sample_number == len(rows):
                print(f"[EXTENDED] tick={tick} state={sample_number}/{len(rows)}", flush=True)
        for method in methods:
            same=[]; different=[]
            for i in range(len(rows)):
                for j in range(i+1,len(rows)):
                    value=similarity(states[i],states[j],method); relation="same_class" if rows[i]["label"]==rows[j]["label"] else "different_class"; (same if relation=="same_class" else different).append(value); pair_out.append({"method":method,"tick":tick,"relation":relation,"similarity":value,"sample_a":rows[i]["sample_id"],"sample_b":rows[j]["sample_id"]})
            tick_out.append({"method":method,"tick":tick,"same_mean":float(np.mean(same)),"different_mean":float(np.mean(different)),"same_minus_different":float(np.mean(same)-np.mean(different)),"seconds":time.time()-started})
        print(f"[EXTENDED] tick={tick} done seconds={time.time()-started:.1f}",flush=True)
    for name,data in (("pairwise_similarity_sparse.csv",pair_out),("tick_summary_sparse.csv",tick_out)):
        with (a.output_dir/name).open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    (a.output_dir/"run_manifest_sparse.json").write_text(json.dumps({"methods":methods,"ticks":ticks,"density":a.density},indent=2),encoding="utf-8")


if __name__ == "__main__": main()
