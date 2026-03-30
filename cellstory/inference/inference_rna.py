import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse
import json
import os

def append_to_obsm(adata, obsm_key, embedding):
    adata.obsm[obsm_key] = embedding


def plot_umap_raw(
    adata,
    umap_png,
    ax,
    umap_title=None,
    n_neighbors=30,
    key="X",
    layer_key="counts",
    celltype_key="cell_type",
    umap_key="X_umap",
    seed=42,
):
    neighbors_key = f"{key}_neighbors"
    leiden_key = f"{key}_leiden"

    # use copied raw counts layer
    adata.X = adata.layers[layer_key].copy()
    # Normalizing to median total counts
    sc.pp.normalize_total(adata)
    # Logarithmize the data
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    sc.tl.pca(adata)
    sc.pp.neighbors(
        adata, n_neighbors=30, n_pcs=50, use_rep=None, key_added=neighbors_key
    )
    sc.tl.leiden(adata, key_added=leiden_key, neighbors_key=neighbors_key)
    sc.tl.umap(adata, neighbors_key=neighbors_key)
    # plot single umap & return
    umap_fig = sc.pl.umap(
        adata,
        color=[celltype_key],
        title=umap_title,
        neighbors_key=neighbors_key,
        return_fig=True,
        show=False,
    )
    # save umap
    umap_fig.savefig(umap_png, bbox_inches="tight")
    # plot on ax
    sc.pl.umap(
        adata,
        color=[celltype_key],
        title=umap_title,
        neighbors_key=neighbors_key,
        return_fig=False,
        show=False,
        ax=ax,
    )
    metrics = kmeans_umap(
        adata, umap_key=umap_key, celltype_key=celltype_key, seed=seed
    )
    # return values
    return umap_fig, metrics


def plot_umap_embed(
    adata,
    umap_png,
    ax,
    umap_title=None,
    n_neighbors=30,
    key="cellstory_rna",
    celltype_key="cell_type",
    umap_key="X_umap",
    seed=42,
):
    rep_key = key
    neighbors_key = f"{key}_neighbors"
    leiden_key = f"{key}_leiden"
    sc.pp.neighbors(adata, n_neighbors=30, use_rep=rep_key, key_added=neighbors_key)
    sc.tl.leiden(adata, key_added=leiden_key, neighbors_key=neighbors_key)
    sc.tl.umap(adata, neighbors_key=neighbors_key)
    # plot single umap & return
    umap_fig = sc.pl.umap(
        adata,
        color=[celltype_key],
        title=umap_title,
        neighbors_key=neighbors_key,
        return_fig=True,
        show=False,
    )
    # save umap
    umap_fig.savefig(umap_png, bbox_inches="tight")
    # plot on ax
    sc.pl.umap(
        adata,
        color=[celltype_key],
        title=umap_title,
        neighbors_key=neighbors_key,
        return_fig=False,
        show=False,
        ax=ax,
    )
    metrics = kmeans_umap(
        adata, umap_key=umap_key, celltype_key=celltype_key, seed=seed
    )
    # return values
    return umap_fig, metrics


def kmeans_umap(adata, umap_key="X_umap", celltype_key="cell_type", seed=42):
    n_clusters = adata.obs[celltype_key].nunique()
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    labels_pred = kmeans.fit_predict(adata.obsm[umap_key])


    labels_true = adata.obs[celltype_key]
    # silhouette = silhouette_score(adata.obsm[umap_key], labels_pred, metric="euclidean")
    ari = adjusted_rand_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)

    metrics = {"ARI": ari, "NMI": nmi}
    return metrics


def generate_rna_metrics(args, inferred_adata):
    rna_h5ad_stem = args.rna_h5ad.stem
    embed_h5ads = {}
    compare_umap_pngs = {}
    umap_metric_tsvs = {}
    metric_data = []

    # Define obsm keys based on the task
    if args.task == "rnaatacmlm":
        obsm_keys = {
            'rna': 'obsm_rna',
            'combined': 'obsm_atac_rna',
            'atac': 'obsm_atac'
        }
    else:
        obsm_keys = {
            'default': args.obsm_key
        }


    for key, obsm_key in obsm_keys.items():
        embed_h5ad = args.dirpath / f"{rna_h5ad_stem}.{obsm_key}.h5ad"
        compare_umap_png = args.dirpath / f"{rna_h5ad_stem}.umap.compare.{obsm_key}.png"
        raw_umap_png = args.dirpath / f"{rna_h5ad_stem}.umap.raw.png"
        embed_umap_png = args.dirpath / f"{rna_h5ad_stem}.umap.{obsm_key}.png"
        umap_metric_tsv = args.dirpath / f"{rna_h5ad_stem}.umap.metrics.{obsm_key}.tsv"

        fig, ((embed_ax, raw_ax)) = plt.subplots(
            2, 1, figsize=(6.4, 9.6), gridspec_kw=dict(wspace=0.5)
        )
        # Embedding UMAP
        umap_fig_embed, embed_metrics = plot_umap_embed(
            inferred_adata,
            umap_png=str(embed_umap_png),
            ax=embed_ax,
            umap_title=obsm_key,
            n_neighbors=30,
            key=obsm_key,
            celltype_key="cell_type",
            umap_key="X_umap",
            seed=args.seed,
        )
        # Raw counts UMAP
        umap_fig_raw, raw_metrics = plot_umap_raw(
            inferred_adata,
            umap_png=str(raw_umap_png),
            ax=raw_ax,
            umap_title="raw_counts",
            n_neighbors=30,
            key="X",
            layer_key="counts",
            celltype_key="cell_type",
            umap_key="X_umap",
            seed=args.seed,
        )
        # Save compare figure
        fig.suptitle(f"{obsm_key} vs raw_counts")
        fig.savefig(str(compare_umap_png), bbox_inches="tight")

        # Save metric df
        metric_df = pd.DataFrame(
            [raw_metrics, embed_metrics], index=["raw_counts", obsm_key]
        )
        metric_df.to_csv(umap_metric_tsv, sep="\t")

        # Save h5ad for each embedding
        inferred_adata.write_h5ad(str(embed_h5ad))

        embed_h5ads[obsm_key] = embed_h5ad
        compare_umap_pngs[obsm_key] = compare_umap_png
        umap_metric_tsvs[obsm_key] = umap_metric_tsv
        metric_data.append(metric_df)

    return embed_h5ads, compare_umap_pngs, umap_metric_tsvs, metric_data


def rna_perturbation_metrics(args, preds, reals):
    adata = sc.read_h5ad(args.rna_h5ad)
    ctrl_adata = sc.read(args.train_h5ad)
    ctrl_adata = ctrl_adata[ctrl_adata.obs["condition"] == "ctrl"]

    gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))
    gene2idx = {x: it for it, x in enumerate(adata.var.gene_name)}
    mse_de_results = []
    pearson_de_results = []
    pearson_delta_results = []
    pearson_delta_de_results = []
    for perturb in preds.keys():

        genede_idx20 = [
            gene2idx[gene_raw2id[i]]
            for i in adata.uns['top_non_zero_de_20'][f'{adata.obs["cell_type"][0]}_{perturb}_1+1']
        ]
        predsmeans = np.mean(preds[perturb], axis=0)
        realmeans = np.mean(reals[perturb], axis=0)
        predsmeans_de20 = np.mean(preds[perturb][:, genede_idx20], axis=0)
        realmeans_de20 = np.mean(reals[perturb][:, genede_idx20], axis=0)

        ctrl_means = np.mean(ctrl_adata.X.toarray(), axis=0)
        ctrl_means_de20 = np.mean(ctrl_adata.X.toarray()[:, genede_idx20], axis=0)

        mse_value_de = mse(predsmeans_de20, realmeans_de20)
        pear_de = pearsonr(predsmeans_de20, realmeans_de20)[0]
        if np.isnan(pear_de):
            pear_de = 0

        pear_delta = pearsonr(predsmeans - ctrl_means, realmeans - ctrl_means)[0]
        pear_de_delta = pearsonr(predsmeans_de20 - ctrl_means_de20, realmeans_de20 - ctrl_means_de20)[0]

        mse_de_results.append(mse_value_de)
        pearson_de_results.append(pear_de)
        pearson_delta_results.append(pear_delta)
        pearson_delta_de_results.append(pear_de_delta)

    avg_mse = np.mean(mse_de_results) if mse_de_results else 0
    avg_pearson = np.mean(pearson_de_results) if pearson_de_results else 0

    avg_pearson_delta = np.mean(pearson_delta_results) if pearson_delta_results else 0
    avg_pearson_delta_de = np.mean(pearson_delta_de_results) if pearson_delta_de_results else 0

    with open(os.path.join(args.dirpath, 'metrics_mse_results.json'), 'w') as f:
        json.dump([float(x) for x in mse_de_results], f, indent=4)
    with open(os.path.join(args.dirpath, 'metrics_pearson_results.json'), 'w') as f:
        json.dump([float(x) for x in pearson_de_results], f, indent=4)
    with open(os.path.join(args.dirpath, 'metrics_pearson_delta_results.json'), 'w') as f:
        json.dump([float(x) for x in pearson_delta_results], f, indent=4)
    with open(os.path.join(args.dirpath, 'metrics_pearson_delta_de_results.json'), 'w') as f:
        json.dump([float(x) for x in pearson_delta_de_results], f, indent=4)
    return avg_mse, avg_pearson, avg_pearson_delta, avg_pearson_delta_de