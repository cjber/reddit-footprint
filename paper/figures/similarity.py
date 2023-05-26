import geopandas as gpd
import matplotlib.pyplot as plt
import polars as pl
from sklearn.metrics.pairwise import cosine_similarity

from src.common.utils import Paths


def plot_similarity(region_embeddings: pl.DataFrame):
    cosine_sim = cosine_similarity(list(region_embeddings["embeddings"]))
    hm_frame = pl.concat(
        [
            pl.from_pandas(region_embeddings[["RGN21NM"]]),
            pl.DataFrame(cosine_sim, schema=region_embeddings["RGN21NM"].to_list()),
        ],
        how="horizontal",
    )
    hm_frame = hm_frame.with_columns(pl.sum(pl.exclude("RGN21NM")).alias("Total")).sort(
        "Total", descending=True
    )

    fig, ax = plt.subplots(3, 4, figsize=(10, 10))
    plt.tight_layout()
    axs = ax.flatten()

    for idx, i in enumerate(hm_frame.partition_by(by="RGN21NM")):
        ax = axs[idx]
        name = i["RGN21NM"][0]
        df_plt = region_embeddings.merge(
            i.to_pandas().set_index("RGN21NM").T.reset_index(),
            left_on="RGN21NM",
            right_on="index",
        )
        df_plt.plot(
            column=df_plt[name],
            cmap="Reds",
            edgecolor="lightgrey",
            linewidth=0.5,
            vmin=cosine_sim.min(),
            vmax=cosine_sim.max(),
            ax=ax,
        )
        (
            df_plt[df_plt[name] == df_plt[name].max()].plot(
                color="green",
                ax=ax,
                edgecolor="white",
                linewidth=0.5,
            )
        )
        ax.set_axis_off()
        ax.set_title(name, y=0.95)
    region_embeddings.merge(
        hm_frame[["RGN21NM", "Total"]].to_pandas(), on="RGN21NM"
    ).plot(
        column="Total",
        cmap="Reds",
        edgecolor="lightgrey",
        linewidth=0.5,
        vmin=hm_frame["Total"].min(),
        vmax=hm_frame["Total"].max(),
        ax=axs[-1],
    )
    axs[-1].set_axis_off()
    axs[-1].set_title("Total", y=0.95)


if __name__ == "__main__":
    en_regions = gpd.read_parquet("./data/processed/en_regions.parquet")
    en_regions["geometry"] = en_regions.simplify(1000)
    region_embeddings = pl.read_parquet(
        Paths.PROCESSED / "region_embeddings.parquet"
    ).to_pandas()
    region_embeddings = en_regions.merge(region_embeddings, on="RGN21NM")

    plot_similarity(region_embeddings)
    plt.show()
