import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import umap
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA

from src.common.utils import Const, filtered_annotation, process_outs


def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = sum(
            1 if child_idx < n_samples else counts[child_idx - n_samples]
            for child_idx in merge
        )
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    dendrogram(linkage_matrix, **kwargs)


def app_dendogram(df):
    embeddings = df["embeddings"].to_list()
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    umap2 = umap.UMAP(n_components=2, random_state=Const.SEED)
    pca_vecs = umap2.fit_transform(embeddings)
    df["vecx"] = pca_vecs[:, 0]
    df["vecy"] = pca_vecs[:, 1]

    clustering_model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

    clustering_model.fit(pca_vecs)
    plot_dendrogram(clustering_model, truncate_mode="level", p=3)


def process_embeddings(df):
    embeddings = df["embeddings"].to_list()
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    umap2 = umap.UMAP(n_components=2, random_state=Const.SEED)
    pca_vecs = umap2.fit_transform(embeddings)
    df["vecx"] = pca_vecs[:, 0]
    df["vecy"] = pca_vecs[:, 1]

    clustering_model = AgglomerativeClustering(n_clusters=3)
    clustering_model.fit(pca_vecs)
    df["cluster"] = clustering_model.labels_

    pca_1 = PCA(n_components=1, random_state=Const.SEED)
    pca_result = pca_1.fit_transform(embeddings)
    df["pca"] = pca_result[:, 0]
    return df


def plt_place_vectors(df: pl.DataFrame, rgn) -> None:
    df = process_embeddings(df)

    fig = plt.figure(figsize=(8, 6))
    custom_cmap = ListedColormap(sns.color_palette("viridis_r"))

    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1])
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("(a)")

    sns.scatterplot(
        data=df,
        x="vecx",
        y="vecy",
        hue="cluster",
        palette=custom_cmap,
        edgecolor=None,
        s=5,
        ax=ax,
        legend=False,
    )
    plt.axis("off")

    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("(b)")
    sns.histplot(
        data=df,
        x="RGN22NM",
        stat="proportion",
        hue="cluster",
        multiple="fill",
        ax=ax,
        palette=custom_cmap,
        legend=False,
    )
    plt.xticks(rotation=45, ha="right")

    ax = fig.add_subplot(gs[:, 1])
    ax.set_title("(c)")
    df.plot(
        column="cluster",
        ax=ax,
        cmap=custom_cmap,
        legend=True,
        legend_kwds={"loc": "lower center", "ncols": 3, "frameon": False},
        categorical=True,
        edgecolor="face",
    )
    rgn.boundary.plot(
        color=None,
        edgecolor="black",
        linewidth=0.5,
        alpha=1,
        ax=ax,
    )

    filtered_annotation(
        df,
        [
            "Glasgow City",
            "City of Edinburgh",
            "Cardiff",
            # "Pembrokeshire",
            "City of London",
        ],
        ax,
    )

    plt.axis("off")


if __name__ == "__main__":
    _, regions, _, _, lad_embeddings = process_outs()
    x = process_outs()
    x[0].filter(pl.col("text").str.contains("gonnae"))['RGN22NM'].value_counts()

    plt_place_vectors(lad_embeddings, regions)
    plt.show()
    app_dendogram(lad_embeddings)
    plt.show()
