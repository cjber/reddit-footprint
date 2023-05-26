import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib import gridspec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.common.utils import SEED, Paths


def plot_place_vectors(region_embeddings: pl.DataFrame, en_regions, basemap) -> None:
    kmeans = KMeans(n_clusters=5, random_state=SEED, n_init="auto")
    clusters = kmeans.fit_predict(list(region_embeddings["embeddings"]))

    pca = PCA(n_components=2, random_state=SEED)
    pca_vecs = pca.fit_transform(list(region_embeddings["embeddings"]))
    x0 = pca_vecs[:, 0]
    x1 = pca_vecs[:, 1]

    pca = PCA(n_components=1, random_state=SEED)
    pca_vecs = pca.fit_transform(list(region_embeddings["embeddings"]))

    region_embeddings = pd.concat(
        [
            region_embeddings,
            pd.DataFrame({"x0": x0, "x1": x1, "pca": pca_vecs.flatten()}),
        ],
        axis=1,
    )
    region_embeddings = pd.concat(
        [region_embeddings, pd.DataFrame({"cluster": clusters})], axis=1
    )
    region_embeddings["cluster"] = region_embeddings["cluster"] + 1

    _ = plt.figure(figsize=(6.5, 3))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    gs.update(wspace=-0.05, hspace=0)
    ax1.set_title("(A)")
    ax2.set_title("(B)")

    ax1.set_xlabel("X0")
    ax1.set_ylabel("X1")
    sns.scatterplot(
        data=region_embeddings,
        x="x0",
        y="x1",
        hue="cluster",
        palette="viridis",
        edgecolor="lightgrey",
        s=20,
        ax=ax1,
    )
    for idx, row in region_embeddings.iterrows():
        ax1.text(row["x0"] + 0.003, row["x1"], row["RGN21NM"], fontsize=8)

    handles, labels = ax1.get_legend_handles_labels()
    for ha in handles:
        ha.set_edgecolor("lightgrey")
        ha.set_linewidth(0.5)
    sns.move_legend(
        obj=ax1,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=5,
        title=None,
        frameon=False,
        markerscale=0.8,
        handles=handles,
    )

    ax1.set(ylabel=None, xlabel=None)
    ax1.set(yticklabels=[], xticklabels=[])
    ax1.tick_params(left=False, bottom=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    # ax1.spines['bottom'].set_visible(False)
    ax1.spines["left"].set_visible(False)

    basemap.plot(ax=ax2, color="lightgrey")
    region_embeddings.plot(
        column="cluster",
        ax=ax2,
        cmap="viridis",
        edgecolor="lightgrey",
        linewidth=0.5,
    )
    # ax2.set(yticklabels=[], xticklabels=[], ylabel=None, xlabel=None)
    ax2.set_xlim(left=10_000)
    plt.axis("off")


if __name__ == "__main__":
    basemap = gpd.read_parquet(Paths.RAW / "ukpoly-2023_03_03.parquet")
    region_embeddings = pl.read_parquet(
        Paths.PROCESSED / "region_embeddings.parquet"
    ).to_pandas()
    en_regions = gpd.read_parquet("./data/processed/en_regions.parquet")
    en_regions["geometry"] = en_regions.simplify(1000)

    region_embeddings = gpd.GeoDataFrame(
        region_embeddings.merge(en_regions, on="RGN21NM")
    )

    plot_place_vectors(region_embeddings, en_regions, basemap)
    plt.show()
