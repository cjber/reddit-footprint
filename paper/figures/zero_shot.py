import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import polars as pl
import seaborn.objects as so
from seaborn import axes_style

from src.common.utils import Const, Paths, process_outs


def plt_zs_lad(zero_shot_path, lad):
    places = (
        pl.scan_parquet(zero_shot_path)
        .unique(["text", "scores"])
        .group_by(["LAD22NM", "labels"])
        .mean()
        .sort("scores", descending=True)
        .unique("LAD22NM")
        .collect()
    )

    p = places.to_pandas().merge(lad, left_on="LAD22NM", right_on="LAD22NM")
    p = gpd.GeoDataFrame(p, geometry="geometry")

    fig, ax = plt.subplots(figsize=(8, 6))
    p.plot("scores", legend=True, ax=ax)
    p.plot("labels", legend=True, ax=ax)
    plt.show()


def plt_zero_shot(zero_shot_path):
    places = (
        pl.scan_parquet(zero_shot_path)
        .unique(["text", "scores"])
        .group_by(["RGN21NM", "LAD22NM", "labels"])
        .mean()
        # .with_columns(
        #     pl.col("labels").replace(
        #         {"English": "E", "British": "B", "Scottish": "S", "Welsh": "W"}
        #     ),
        # )
        .collect(streaming=True)
    )
    idx = list(
        enumerate(
            places.filter(pl.col("labels") == "British")
            .group_by("RGN21NM")
            .mean()
            .sort("scores", descending=True)["RGN21NM"]
            .to_list()
        )
    )
    places = places.join(
        pl.DataFrame(idx), left_on="RGN21NM", right_on="column_1"
    ).sort("column_0")

    fig, ax = plt.subplots()
    ax.hlines(
        y=[idx - 0.5 for idx, _ in enumerate(places["RGN21NM"].unique())],
        xmin=0,
        xmax=places["scores"].max(),
        linewidth=1,
        color="lightgrey",
        linestyle=":",
    )
    ax.hlines(
        y=8.5,
        xmin=0,
        xmax=places["scores"].max(),
        linewidth=1,
        color="grey",
        linestyle="dashed",
    )
    _ = (
        so.Plot(
            places.to_pandas(),
            x="scores",
            y="RGN21NM",
            text="labels",
            color="labels",
        )
        .add(
            so.Range(linewidth=1, color="black"),
            so.Est(seed=0, errorbar="se"),
            # so.Shift(y=0.1),
            so.Dodge(),
        )
        .add(
            # so.Text(
            #     artist_kws={"fontweight": "bold"}, offset=0, valign="center", fontsize=8
            # ),
            so.Dot(edgewidth=1, edgecolor="black", artist_kws={"zorder": 10}),
            so.Agg(),
            so.Dodge(),
        )
        .label(x="Confidence Value", y="")
        .theme({**axes_style("ticks")})
        .on(ax)
        .layout(engine="tight")
        .scale(
            color={
                "Scottish": "#1f77b4",
                "English": "#ff7f0e",
                "British": "#2ca02c",
                "Welsh": "#d62728",
            },
        )
        .plot()
    )
    legend_contents = _._legend_contents
    handles = []
    labels = []
    blank_handle = mpl.patches.Patch(alpha=0, linewidth=0, visible=False)
    for legend_content in legend_contents:
        handles.append(blank_handle)
        handles.extend(legend_content[1])
        labels.append(legend_content[0][0])
        labels.extend(legend_content[2])
    ax.legend(handles[6:], labels[6:], frameon=False)
    fig.legends = []
    _.show()


if __name__ == "__main__":
    plt_zero_shot(Paths.PROCESSED / "places_zero_shot.parquet")
    plt.show()
