# CTM sync visualization contract

## Claim

After sequential AE updates, the frozen CTM probe records changes in its selected-neuron synchronisation statistics for fixed old MNIST samples. These changes are observable; they do not by themselves establish forgetting or a causal mechanism.

## Source representation

For the audited `first-last` setup, `sync_out` selects the first 32 CTM neurons and `sync_action` selects the last 32. Each 528-value vector is the upper triangle of the symmetric 32x32 exponentially weighted activation-product statistic (`32*33/2=528`). Reconstruct by assigning values to both `(i,j)` and `(j,i)` using PyTorch `triu_indices`; do not call the result a correlation matrix. CTM source commit: `4a6c9c3a7fb5dc4bca6381cc7883a3b9252c6466`.

## Figure set

1. One fixed old sample: initial vs after class 4 vs signed difference, at the final saved tick, for `out` and `action`.
2. Same sample across saved ticks `1,2,4,8,12,16,20`, showing the matrix evolution for both outputs.
3. Class-mean matrices for old labels 0/1/2, before and after class 4, plus signed differences.
4. Cosine similarity among class-mean synchronisation vectors before and after class 4.
5. Descriptive per-seed association of sync drift with the old-sample reconstruction-MSE increase. Six points only; no fitted trend or significance claim.

## Evidence and limits

Use the same probe IDs and tick IDs across checkpoints. Use one shared color range for before/after panels within each output; center difference panels at zero. Figures show one representative sample or class means and must retain per-seed/source-data CSVs. The fixed old set is MNIST test data. No GPU is required for plotting; the saved traces and labels are sufficient.
