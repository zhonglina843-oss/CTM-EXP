#!/usr/bin/env python3
"""CPU benchmark for the remaining CTM graph-comparison methods.

Exact GED/GW are not practical on 512-node graphs. Their outputs are named
ged_approx/gw_approx and the approximation is recorded in method_notes.md.
DeltaCon uses sparse linear solves with random probes instead of dense inverse.
"""

from __future__ import annotations

import argparse
import csv
import math
import resource
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import factorized


METHODS = (
    "community_nmi",
    "community_ari",
    "delta_con",
    "delta_con_attr",
    "ged_approx",
    "wl_kernel",
    "gw_approx",
    "dynamic_trajectory",
)


@dataclass
class Sample:
    sample_id: str
    label: int
    class_sample_index: int
    label_name: str
    active_ids: np.ndarray
    matrices: np.ndarray


def read_labels(path: Path, samples_per_class: int) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["label"] = int(row["label"])
        row["class_sample_index"] = int(row.get("class_sample_index", 0))
    rows.sort(key=lambda row: (row["label"], row["class_sample_index"]))
    selected, counts = [], {}
    for row in rows:
        if counts.get(row["label"], 0) < samples_per_class:
            selected.append(row)
            counts[row["label"]] = counts.get(row["label"], 0) + 1
    if not selected or min(counts.values()) < samples_per_class:
        raise ValueError(f"expected {samples_per_class} samples per class, got {counts}")
    return selected


def load_samples(result_dir: Path, rows: list[dict]) -> list[Sample]:
    samples = []
    for row in rows:
        directory = result_dir / "samples" / row["sample_id"]
        matrices = np.asarray(np.load(directory / "S_full_active_all_ticks.npz")["matrices"], dtype=float)
        active = np.asarray(np.load(directory / "active_neuron_indices.npy"), dtype=int)
        if matrices.ndim != 3 or matrices.shape[1] != active.size:
            raise ValueError(f"invalid files for {row['sample_id']}: {matrices.shape}, {active.shape}")
        samples.append(Sample(row["sample_id"], row["label"], row["class_sample_index"],
                              row.get("label_name", str(row["label"])), active, matrices))
    return samples


def top_graph(matrix: np.ndarray, active: np.ndarray, density: float, component: str):
    n = matrix.shape[0]
    ii, jj = np.triu_indices(n, 1)
    weights = np.maximum(matrix[ii, jj], 0.0)
    valid = np.isfinite(weights) & (weights > 0)
    ii, jj, weights = ii[valid], jj[valid], weights[valid]
    k = min(max(1, int(round(n * (n - 1) / 2 * density))), weights.size)
    if k:
        keep = np.argpartition(weights, -k)[-k:]
        ii, jj, weights = ii[keep], jj[keep], weights[keep]
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_weighted_edges_from((int(a), int(b), float(w)) for a, b, w in zip(ii, jj, weights))
    if component == "lcc" and graph.number_of_nodes():
        graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
    edges = {tuple(sorted((int(active[a]), int(active[b])))): float(data["weight"])
             for a, b, data in graph.edges(data=True)}
    return graph, edges


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den else 0.0


def edge_cosine(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    return cosine(np.asarray([a.get(k, 0.0) for k in keys]), np.asarray([b.get(k, 0.0) for k in keys]))


def weighted_l1(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    return float(sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys))


def partition_scores(left: list[set], right: list[set]) -> tuple[float, float]:
    universe = sorted(set().union(*left, *right)) if left or right else []
    if not universe:
        return 1.0, 1.0
    li = {node: i for i, node in enumerate(universe)}
    a = np.full(len(universe), -1, int)
    b = np.full(len(universe), -1, int)
    for i, group in enumerate(left):
        for node in group:
            a[li[node]] = i
    for i, group in enumerate(right):
        for node in group:
            b[li[node]] = i
    valid = (a >= 0) & (b >= 0)
    a, b = a[valid], b[valid]
    if not a.size:
        return 0.0, 0.0
    contingency = np.zeros((a.max() + 1, b.max() + 1), dtype=float)
    for x, y in zip(a, b):
        contingency[x, y] += 1
    n = contingency.sum()
    pi, pj = contingency.sum(1), contingency.sum(0)
    mi = sum(v / n * math.log((v * n) / (pi[i] * pj[j]))
             for (i, j), v in np.ndenumerate(contingency) if v > 0)
    h1 = -sum(x / n * math.log(x / n) for x in pi if x)
    h2 = -sum(x / n * math.log(x / n) for x in pj if x)
    nmi = (2 * mi / (h1 + h2)) if h1 + h2 else 1.0
    comb2 = lambda x: x * (x - 1) / 2
    nij = sum(comb2(x) for x in contingency.flat)
    expected = sum(comb2(x) for x in pi) * sum(comb2(x) for x in pj) / comb2(n) if n > 1 else 0
    maximum = 0.5 * (sum(comb2(x) for x in pi) + sum(comb2(x) for x in pj))
    ari = (nij - expected) / (maximum - expected) if maximum != expected else 1.0
    return float(nmi), float(ari)


def wl_features(graph: nx.Graph, active: np.ndarray, rounds: int = 3) -> Counter:
    labels = {int(active[node]): str(graph.degree(node)) for node in graph.nodes()}
    for _ in range(rounds):
        labels = {node: labels[node] + "|" + ",".join(sorted(labels[int(active[n])] for n in graph.neighbors(local)))
                  for local, node in ((node, int(active[node])) for node in graph.nodes())}
    return Counter(labels.values())


def counter_cosine(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    return cosine(np.asarray([a.get(k, 0) for k in keys], float), np.asarray([b.get(k, 0) for k in keys], float))


def graph_metric_histogram(graph: nx.Graph, active: np.ndarray, sample_nodes: int = 32) -> np.ndarray:
    nodes = list(graph.nodes())[:sample_nodes]
    if not nodes:
        return np.zeros(2)
    lengths = dict(nx.all_pairs_shortest_path_length(graph))
    distances = [lengths[a].get(b, len(graph) + 1) for i, a in enumerate(nodes) for b in nodes[i + 1:]]
    degree = np.asarray([graph.degree(a, weight="weight") for a in nodes], float)
    return np.concatenate([np.sort(resample(np.asarray(distances, float), 64)), np.sort(resample(degree, 32))])


def resample(values: np.ndarray, size: int) -> np.ndarray:
    if not values.size:
        return np.zeros(size)
    if values.size == size:
        return values
    return np.interp(np.linspace(0, 1, size), np.linspace(0, 1, values.size), values)


def build_states(samples: list[Sample], tick: int, density: float, component: str):
    states = {}
    for sample in samples:
        graph, edges = top_graph(sample.matrices[tick], sample.active_ids, density, component)
        communities = [set(int(sample.active_ids[node]) for node in group)
                       for group in nx.algorithms.community.greedy_modularity_communities(graph, weight="weight")]
        degree = {int(sample.active_ids[node]): float(value) for node, value in graph.degree(weight="weight")}
        states[sample.sample_id] = {
            "graph": graph, "edges": edges, "communities": communities, "degree": degree,
            "wl": wl_features(graph, sample.active_ids),
            "gw": graph_metric_histogram(graph, sample.active_ids),
        }
    return states


def delta_con_states(samples: list[Sample], states: dict, probes: int, seed: int):
    active_union = sorted(set().union(*(set(sample.active_ids) for sample in samples)))
    index = {node: i for i, node in enumerate(active_union)}
    n = len(active_union)
    rng = np.random.default_rng(seed)
    rhs = rng.choice(np.array([-1.0, 1.0]), size=(n, probes)) / math.sqrt(probes)
    result = {}
    for sample in samples:
        graph = states[sample.sample_id]["graph"]
        rows, cols, data = [], [], []
        degree = defaultdict(float)
        for (a, b), value in states[sample.sample_id]["edges"].items():
            ia, ib = index[a], index[b]
            rows += [ia, ib]
            cols += [ib, ia]
            data += [value, value]
            degree[ia] += value
            degree[ib] += value
        adjacency = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        d = sparse.diags([degree.get(i, 0.0) for i in range(n)])
        system = sparse.eye(n, format="csc") + 0.05 * d - 0.05 * adjacency
        solve = factorized(system)
        affinity = np.asarray(solve(rhs))
        result[sample.sample_id] = {"affinity": affinity, "active_indices": [index[x] for x in sample.active_ids]}
    return result


def compare(left, right, method: str) -> float:
    if method == "community_nmi" or method == "community_ari":
        nmi, ari = partition_scores(left["communities"], right["communities"])
        return nmi if method.endswith("nmi") else ari
    if method == "ged_approx":
        scale = sum(left["edges"].values()) + sum(right["edges"].values()) + 1e-12
        return 1.0 - weighted_l1(left["edges"], right["edges"]) / scale
    if method == "wl_kernel":
        return counter_cosine(left["wl"], right["wl"])
    if method == "gw_approx":
        distance = float(np.linalg.norm(left["gw"] - right["gw"]) / (np.linalg.norm(left["gw"]) + np.linalg.norm(right["gw"]) + 1e-12))
        return 1.0 / (1.0 + distance)
    raise ValueError(method)


def mean_stats(values):
    if not values:
        return {"mean": "", "median": "", "std": "", "n": 0}
    x = np.asarray(values, float)
    return {"mean": float(x.mean()), "median": float(np.median(x)), "std": float(x.std()), "n": int(x.size)}


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_grid(path: Path, rows: list[dict], samples_per_class: int):
    values = defaultdict(dict)
    for row in rows:
        values[row["label"]][row["class_sample_index"]] = row["similarity_to_same_class_mean"]
    write_csv(path, [{"label": label, **{f"sample_{i}": values[label].get(i, "") for i in range(samples_per_class)}}
                    for label in sorted(values)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--subset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ticks", default="1,5,10,15,20,30,40,50")
    parser.add_argument("--density", type=float, default=0.01)
    parser.add_argument("--component", choices=["all", "lcc"], default="lcc")
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument("--delta-probes", type=int, default=8)
    args = parser.parse_args()
    ticks = [int(x) for x in args.ticks.split(",") if x.strip()]
    rows = read_labels(args.subset_dir / "labels.csv", args.samples_per_class)
    samples = load_samples(args.result_dir, rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_rows, tick_rows, sample_rows, runtime_rows, grid_rows = [], [], [], [], defaultdict(list)
    by_label = defaultdict(list)
    for sample in samples:
        by_label[sample.label].append(sample)
    all_states = {}
    for tick_number in ticks:
        start_state = time.perf_counter()
        states = build_states(samples, tick_number - 1, args.density, args.component)
        delta_states = delta_con_states(samples, states, args.delta_probes, 20260920 + tick_number)
        all_states[tick_number] = states
        state_seconds = time.perf_counter() - start_state
        for method in METHODS:
            start = time.perf_counter()
            same, different, per_sample = [], [], defaultdict(lambda: {"same": [], "different": []})
            for i, left_sample in enumerate(samples):
                for right_sample in samples[i + 1:]:
                    if method in ("delta_con", "delta_con_attr"):
                        la, ra = delta_states[left_sample.sample_id], delta_states[right_sample.sample_id]
                        if method == "delta_con":
                            similarity = 1.0 / (1.0 + float(np.linalg.norm(la["affinity"] - ra["affinity"])))
                        else:
                            li, ri = la["active_indices"], ra["active_indices"]
                            common = sorted(set(li) & set(ri))
                            diff = np.asarray([np.linalg.norm(la["affinity"][i] - ra["affinity"][i]) for i in common])
                            similarity = 1.0 / (1.0 + float(diff.mean() if diff.size else 0.0))
                    elif method == "dynamic_trajectory":
                        if tick_number == ticks[0]:
                            similarity = compare(all_states[tick_number][left_sample.sample_id], all_states[tick_number][right_sample.sample_id], "wl_kernel")
                        else:
                            previous = all_states[ticks[ticks.index(tick_number) - 1]]
                            a = weighted_l1(previous[left_sample.sample_id]["edges"], all_states[tick_number][left_sample.sample_id]["edges"])
                            b = weighted_l1(previous[right_sample.sample_id]["edges"], all_states[tick_number][right_sample.sample_id]["edges"])
                            similarity = 1.0 / (1.0 + abs(a - b) / (a + b + 1e-12))
                    else:
                        similarity = compare(states[left_sample.sample_id], states[right_sample.sample_id], method)
                    relation = "same_class" if left_sample.label == right_sample.label else "different_class"
                    (same if relation == "same_class" else different).append(similarity)
                    per_sample[left_sample.sample_id]["same" if relation == "same_class" else "different"].append(similarity)
                    per_sample[right_sample.sample_id]["same" if relation == "same_class" else "different"].append(similarity)
                    pair_rows.append({"method": method, "tick": tick_number, "relation": relation,
                                      "sample_a": left_sample.sample_id, "label_a": left_sample.label,
                                      "sample_b": right_sample.sample_id, "label_b": right_sample.label,
                                      "similarity": similarity})
            sm, dm = mean_stats(same), mean_stats(different)
            tick_rows.append({"method": method, "tick": tick_number, "same_mean": sm["mean"], "same_median": sm["median"], "same_std": sm["std"], "same_n": sm["n"],
                              "different_mean": dm["mean"], "different_median": dm["median"], "different_std": dm["std"], "different_n": dm["n"],
                              "same_minus_different": float(sm["mean"] - dm["mean"]) if sm["mean"] != "" and dm["mean"] != "" else ""})
            for sample in samples:
                sm, dm = mean_stats(per_sample[sample.sample_id]["same"]), mean_stats(per_sample[sample.sample_id]["different"])
                sample_rows.append({"method": method, "tick": tick_number, "sample_id": sample.sample_id, "label": sample.label,
                                    "class_sample_index": sample.class_sample_index, "same_mean": sm["mean"], "different_mean": dm["mean"],
                                    "same_minus_different": float(sm["mean"] - dm["mean"]) if sm["mean"] != "" and dm["mean"] != "" else ""})
            for label, class_samples in by_label.items():
                for sample in class_samples:
                    peers = [peer for peer in class_samples if peer.sample_id != sample.sample_id]
                    if method in ("delta_con", "delta_con_attr"):
                        vals = []
                        for peer in peers:
                            la, ra = delta_states[sample.sample_id], delta_states[peer.sample_id]
                            vals.append(1.0 / (1.0 + float(np.linalg.norm(la["affinity"] - ra["affinity"]))))
                    else:
                        vals = [compare(states[sample.sample_id], states[peer.sample_id], method if method != "dynamic_trajectory" else "wl_kernel") for peer in peers]
                    grid_rows[(method, tick_number)].append({"label": label, "class_sample_index": sample.class_sample_index,
                                                               "similarity_to_same_class_mean": float(np.mean(vals))})
            runtime_rows.append({"method": method, "tick": tick_number, "state_build_seconds": state_seconds,
                                 "method_pairwise_seconds": time.perf_counter() - start, "n_samples": len(samples),
                                 "n_pairs": len(samples) * (len(samples) - 1) // 2,
                                 "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024})
    write_csv(args.output_dir / "pairwise_similarity_extended.csv", pair_rows)
    write_csv(args.output_dir / "tick_summary_extended.csv", tick_rows)
    write_csv(args.output_dir / "sample_similarity_extended.csv", sample_rows)
    write_csv(args.output_dir / "runtime_metrics_extended.csv", runtime_rows)
    for (method, tick), values in grid_rows.items():
        write_grid(args.output_dir / f"grid_{method}_tick-{tick:03d}.csv", values, args.samples_per_class)
    (args.output_dir / "extended_run_manifest.json").write_text(
        "{\n  \"methods\": " + repr(list(METHODS)).replace("'", "\"" ) +
        ",\n  \"ticks\": " + repr(ticks) +
        ",\n  \"ged_definition\": \"weighted edge L1 edit-cost approximation\",\n"
        "  \"gw_definition\": \"degree plus sampled shortest-path histogram approximation\",\n"
        "  \"deltacon_definition\": \"sparse random-probe affinity approximation\"\n}\n", encoding="utf-8")
    print(f"DONE extended samples={len(samples)} ticks={len(ticks)} methods={list(METHODS)} output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
