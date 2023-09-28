import esda
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap
from esda.moran import Moran_Local
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pysal.lib import weights
from sklearn.metrics.pairwise import cosine_similarity

from src.common.utils import Const, filtered_annotation, process_outs

N_COMPONENTS = 2


def process_moran(lad_embeddings):
    embeddings = lad_embeddings["embeddings"].to_list()
    umap_n = umap.UMAP(n_components=N_COMPONENTS, random_state=Const.SEED)
    pca_result = umap_n.fit_transform(embeddings)
    explained = 0
    for n in range(N_COMPONENTS):
        lad_embeddings[f"pca{n}"] = pca_result[:, n]

        w1 = weights.contiguity.Queen.from_dataframe(lad_embeddings)
        w2 = weights.KNN.from_dataframe(lad_embeddings, k=8)
        w1 = weights.set_operations.w_union(w1, w2)
        w1.transform = "R"

        lad_embeddings[f"pca_stdd{n}"] = (
            lad_embeddings[f"pca{n}"] - lad_embeddings[f"pca{n}"].mean()
        )
        lad_embeddings[f"pca_std_lag{n}"] = weights.spatial_lag.lag_spatial(
            w1, lad_embeddings[f"pca_stdd{n}"]
        )

        lisa = Moran_Local(
            lad_embeddings[f"pca_stdd{n}"].astype(float),
            w1,
            keep_simulations=False,
            seed=Const.SEED,
        )

        lad_embeddings[f"lisa_q{n}"] = lisa.q
        lad_embeddings[f"lisa_p{n}"] = lisa.p_sim
        lad_embeddings[f"Is{n}"] = lisa.Is
    return lad_embeddings, w1, explained


def plt_morans(lad_embeddings, w, explained):
    fig, ax = plt.subplots()

    sim = cosine_similarity(
        lad_embeddings["pca_stdd0"].to_numpy().reshape(1, -1),
        lad_embeddings["pca_stdd1"].to_numpy().reshape(1, -1),
    )

    for n, c in zip(range(N_COMPONENTS), ["black", "blue", "red", "yellow"]):
        sns.regplot(
            x=f"pca_stdd{n}",
            y=f"pca_std_lag{n}",
            ci=None,
            data=lad_embeddings,
            color=c,
            marker="x",
            line_kws={"lw": 1},
            scatter_kws={"alpha": 0.5, "s": 10, "linewidths": 1},
        )
    ax.axvline(0, c="k", alpha=0.5, linestyle="--")
    ax.axhline(0, c="k", alpha=0.5, linestyle="--")
    moran1 = esda.moran.Moran(lad_embeddings["pca_stdd0"], w)
    moran2 = esda.moran.Moran(lad_embeddings["pca_stdd1"], w)
    ax.set_title(
        f"Moran's I (1): {moran1.I:.2f}; Moran's I (2): {moran2.I:.2f};"
        f" Cosine Similarity: {sim[0].item():.0%}",
    )
    return moran1, moran2, sim


def plt_lisa(lad_embeddings, n):
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(len(n), len(n) * 3, hspace=0.1, wspace=0.1)

    vmin = lad_embeddings[["pca_stdd0", "pca_stdd1"]].min().min()
    vmax = lad_embeddings[["pca_stdd0", "pca_stdd1"]].max().max()

    _lisa_subplots(fig, lad_embeddings, gs, n[0], "abc")
    _lisa_subplots(fig, lad_embeddings, gs, n[1], "def")


def _lisa_subplots(fig, lad_embeddings, gs, n, s):
    cmap = matplotlib.colormaps["viridis"]

    ax = fig.add_subplot(gs[n, 0])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="2%", pad=0)
    ax.axis("off")
    ax.set_title(f"({s[0]}) UMAP: Dim {n}", pad=-1)
    lad_embeddings.plot(
        column=f"pca_stdd{n}",
        edgecolor="face",
        cmap="viridis",
        ax=ax,
        cax=cax,
        legend=True,
        legend_kwds={"orientation": "horizontal", "pad": 0},
    )

    ax = fig.add_subplot(gs[n, 1])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="2%", pad=0)
    ax.axis("off")
    ax.set_title(f"({s[1]}) Moran's I: Dim {n}", pad=-1)
    lad_embeddings.plot(
        column=f"Is{n}",
        vmax=1,
        vmin=-1,
        edgecolor="face",
        cmap="viridis",
        ax=ax,
        cax=cax,
        legend=True,
        legend_kwds={"orientation": "horizontal", "pad": 0},
    )

    ax = fig.add_subplot(gs[n, 2])
    ax.axis("off")
    ax.set_title(f"({s[2]}) LISA: Dim {n}", pad=-1)
    lad_embeddings.loc[(lad_embeddings[f"lisa_p{n}"] >= 0.05), "cmap"] = np.nan
    lad_embeddings.loc[
        ((lad_embeddings[f"lisa_q{n}"] == 2) | (lad_embeddings[f"lisa_q{n}"] == 4))
        & (lad_embeddings[f"lisa_p{n}"] < 0.05),
        "cmap",
    ] = "HL/LH"
    lad_embeddings.loc[
        (lad_embeddings[f"lisa_q{n}"] == 1) & (lad_embeddings[f"lisa_p{n}"] < 0.05),
        "cmap",
    ] = "HH"
    lad_embeddings.loc[
        (lad_embeddings[f"lisa_q{n}"] == 3) & (lad_embeddings[f"lisa_p{n}"] < 0.05),
        "cmap",
    ] = "LL"

    lad_embeddings.plot(
        "cmap",
        cmap="viridis",
        edgecolor="face",
        categorical=True,
        missing_kwds={
            "color": "lightgrey",
            "label": "_nolegend_",
            "edgecolor": "lightgrey",
        },
        legend=True,
        legend_kwds={
            # "bbox_to_anchor": (0, 0),
            "loc": "lower center",
            "ncols": 3,
            "frameon": False,
            "borderpad": -1,
            "columnspacing": 1,
        },
        ax=ax,
    )
    if s == "abc":
        filtered_annotation(lad_embeddings, ["City of London"], ax)
    else:
        filtered_annotation(lad_embeddings, ["Glasgow City", "City of Edinburgh"], ax)


if __name__ == "__main__":
    _, _, _, _, lad_embeddings = process_outs()
    lad_embeddings, w1, explained = process_moran(lad_embeddings)

    plt_morans(lad_embeddings, w1, explained)
    plt.show()

    plt_lisa(lad_embeddings, [0, 1])
    plt.show()
