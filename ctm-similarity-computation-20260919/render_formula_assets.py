#!/usr/bin/env python3
"""Render README formulas as SVG assets for Markdown viewers without MathJax."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FORMULAS = {
    "01_weighted_jaccard": r"$J_w(a,b)=\frac{\sum_e\min(w_{a,e},w_{b,e})}{\sum_e\max(w_{a,e},w_{b,e})}$",
    "02_edge_cosine": r"$\operatorname{cos}(a,b)=\frac{a^{T}b}{\left\|a\right\|_2\left\|b\right\|_2}$",
    "03_degree_strength": r"$d_i=\sum_j w_{ij},\qquad \operatorname{cos}(d_a,d_b)=\frac{d_a^{T}d_b}{\left\|d_a\right\|_2\left\|d_b\right\|_2}$",
    "04_spectral": r"$\lambda(A)=\operatorname{eigvalsh}(A),\qquad \operatorname{sim}(a,b)=\frac{1}{1+\left\|\lambda_a-\lambda_b\right\|_2/(\left\|\lambda_a\right\|_2+\left\|\lambda_b\right\|_2)}$",
    "05_community": r"$C(G)=\{C_1,\ldots,C_k\},\qquad \operatorname{sim}_{\mathrm{community}}=\operatorname{NMI}(C_a,C_b)\ \mathrm{or}\ \operatorname{ARI}(C_a,C_b)$",
    "06_deltacon": r"$S=(I+\varepsilon D-\varepsilon A)^{-1},\qquad d(G_a,G_b)=\left[\sum_{i,j}\left(\sqrt{S_{ij}^{(a)}}-\sqrt{S_{ij}^{(b)}}\right)^2\right]^{1/2}$",
    "07_deltacon_attr": r"$\operatorname{impact}(v)=\left\| S_{v,:}^{(a)}-S_{v,:}^{(b)}\right\|_2^2$",
    "08_graph_edit_distance": r"$\operatorname{GED}(G_a,G_b)=\min_{E}\operatorname{cost}(E)$",
    "09_graph_kernel_wl": r"$h_v^{(t+1)}=\operatorname{HASH}\!\left(h_v^{(t)},\operatorname{SORT}\{h_u^{(t)}:u\in N(v)\}\right)$",
    "10_gromov_wasserstein": r"$\operatorname{GW}(G_a,G_b)=\min_{\pi}\sum_{i,j,k,l}\left|d_a(i,j)-d_b(k,l)\right|^2\pi_{ik}\pi_{jl}$",
    "11_dynamic_trajectory": r"$\operatorname{TrajSim}=\frac{1}{T}\sum_{t=1}^{T}\operatorname{Sim}(G_{a,t},G_{b,t}),\qquad \Delta G_t=G_t-G_{t-1}$",
}


def main() -> None:
    output_dir = Path(__file__).parent / "formula_assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, formula in FORMULAS.items():
        fig = plt.figure(figsize=(12, 0.7), dpi=180)
        fig.text(0.01, 0.5, formula, fontsize=18, va="center", ha="left")
        fig.savefig(output_dir / f"{name}.svg", format="svg", transparent=True, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
    print(f"rendered {len(FORMULAS)} formulas to {output_dir}")


if __name__ == "__main__":
    main()
