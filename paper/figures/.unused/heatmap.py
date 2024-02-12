import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.patches import Rectangle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from src.common.utils import process_outs


def plt_cosine_heatmap(region_embeddings: pl.DataFrame) -> None:
    region_embeddings = pl.from_pandas(region_embeddings.drop("geometry", axis=1))
    cosine_sim = cosine_similarity(list(region_embeddings["embeddings"]))
    cosine_sim = MinMaxScaler().fit_transform(pl.DataFrame(cosine_sim))

    hm_frame = pl.concat(
        [
            region_embeddings[["RGN22NM"]],
            pl.DataFrame(cosine_sim, schema=region_embeddings["RGN22NM"].to_list()),
        ],
        how="horizontal",
    ).sort("RGN22NM")
    hm_frame = hm_frame.select(["RGN22NM"] + hm_frame["RGN22NM"].to_list())
    lbls = hm_frame["RGN22NM"].apply(lambda x: x.title())

    hm_frame = hm_frame.to_pandas()

    _, ax = plt.subplots(1, figsize=(7, 5))
    sns.heatmap(
        hm_frame.drop("RGN22NM", axis=1),
        xticklabels=lbls,
        yticklabels=lbls,
        linewidth=1,
        square=True,
        linecolor="black",
        cbar=False,
        ax=ax,
    )
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    plt.minorticks_off()
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.tick_params(axis="y", direction="out", pad=2)
    ax.tick_params(axis="x", direction="out", pad=2)

    row_max = hm_frame.replace(hm_frame.max(), 0).drop("RGN22NM", axis=1).idxmax(axis=1)
    for row, _ in enumerate(hm_frame["RGN22NM"]):
        position = hm_frame.drop("RGN22NM", axis=1).columns.get_loc(row_max[row])
        ax.add_patch(
            Rectangle(
                (position + 0.1, row + 0.1),
                0.8,
                0.8,
                fill=False,
                edgecolor="green",
                lw=2,
                alpha=0.5,
            )
        )

    row_min = hm_frame.drop("RGN22NM", axis=1).idxmin(axis=1)
    for row, _ in enumerate(hm_frame["RGN22NM"]):
        position = hm_frame.drop("RGN22NM", axis=1).columns.get_loc(row_min[row])
        ax.add_patch(
            Rectangle(
                (position + 0.1, row + 0.1),
                0.8,
                0.8,
                fill=False,
                edgecolor="red",
                lw=2,
                alpha=0.5,
            )
        )
    plt.show()


if __name__ == "__main__":
    _, _, _, region_embeddings, _ = process_outs()
    plt_cosine_heatmap(region_embeddings)
