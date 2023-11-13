import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import scale

from src.common.utils import process_outs


def plt_similarity(region_embeddings: pl.DataFrame):
    cosine_sim = cosine_similarity(list(region_embeddings["embeddings"]))
    cosine_sim[np.isclose(cosine_sim, 1)] = np.nan
    cosine_sim = scale(cosine_sim.flatten()).reshape((11, 11))
    hm_frame = pl.concat(
        [
            pl.from_pandas(region_embeddings[["RGN22NM"]]),
            pl.DataFrame(cosine_sim, schema=region_embeddings["RGN22NM"].to_list()),
        ],
        how="horizontal",
    ).fill_nan(0)
    hm_frame = hm_frame.with_columns(
        (pl.sum_horizontal(cs.all() - cs.string()) / 11).alias("Mean")
    ).sort("Mean", descending=True)

    fig = plt.figure(figsize=(8, 10))
    gs = gridspec.GridSpec(3, 4, hspace=0.1, wspace=0.1)
    axs = list(gs)

    plt.tight_layout()

    vmin = -2.5
    vmax = 2.5

    for idx, i in enumerate(hm_frame.partition_by(by="RGN22NM")):
        ax = plt.subplot(axs[idx])
        name = i["RGN22NM"][0]
        df_plt = region_embeddings.merge(
            i.to_pandas().set_index("RGN22NM").T.reset_index(),
            left_on="RGN22NM",
            right_on="index",
        )

        custom_cmap = sns.color_palette("viridis", as_cmap=True)
        df_plt.plot(
            column=df_plt[name],
            cmap=custom_cmap,
            edgecolor="face",
            vmin=vmin,
            vmax=vmax,
            ax=ax,
        )
        df_plt[df_plt[name] == 0].plot(
            color="red",
            ax=ax,
            edgecolor="red",
            # alpha=0.5,
        )
        ax.set_axis_off()
        ax.set_title(name, y=0.95)

    axs = plt.subplot(axs[-1])
    region_embeddings.merge(
        hm_frame[["RGN22NM", "Mean"]].to_pandas(), on="RGN22NM"
    ).plot(
        column="Mean",
        cmap=custom_cmap,
        edgecolor="face",
        vmin=vmin,
        vmax=vmax,
        ax=axs,
    )
    axs.set_axis_off()
    axs.set_title("Mean", y=0.95)

    cbar_ax = fig.add_axes([0.12, 0.04, 0.79, 0.01])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(None, cax=cbar_ax, location="bottom", norm=norm)
    breaks = [
        vmin,
        vmin + 0.5,
        vmin + 1,
        vmin + 1.5,
        vmin + 2,
        0.0,
        vmax - 2,
        vmax - 1.5,
        vmax - 1,
        vmax - 0.5,
        vmax,
    ]
    cbar.ax.set_xticks(breaks)
    cbar.ax.set_xticklabels(breaks)


if __name__ == "__main__":
    _, _, _, region_embeddings, _ = process_outs()

    plt_similarity(region_embeddings)
    plt.show()
