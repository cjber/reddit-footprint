import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import umap
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from src.common.utils import Const, process_outs


def process_embeddings(df):
    embeddings = df["embeddings"].to_list()
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5)
    clustering_model.fit(embeddings)
    df["cluster"] = clustering_model.labels_

    umap2 = umap.UMAP(n_components=2, random_state=Const.SEED)
    pca_vecs = umap2.fit_transform(embeddings)
    df["vecx"] = pca_vecs[:, 0]
    df["vecy"] = pca_vecs[:, 1]

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
    ax.set_title("(A)")

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
    ax.set_title("(B)")
    sns.histplot(
        data=df,
        x="RGN21NM",
        stat="proportion",
        hue="cluster",
        multiple="fill",
        ax=ax,
        palette=custom_cmap,
        legend=False,
    )
    plt.xticks(rotation=45, ha="right")

    ax = fig.add_subplot(gs[:, 1])
    ax.set_title("(C)")
    df.plot(
        column="cluster",
        ax=ax,
        cmap=custom_cmap,
        legend=True,
        legend_kwds={"loc": "lower center", "ncols": 3, "frameon": False},
        categorical=True,
        edgecolor="face",
    )
    rgn.boundary.simplify(1000).plot(
        color=None,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.2,
        ax=ax,
    )
    plt.axis("off")


if __name__ == "__main__":
    _, regions, _, _, lad_embeddings = process_outs()

    plt_place_vectors(lad_embeddings, regions)
    plt.show()
