import geopandas as gpd
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
        .with_columns(
            pl.col("labels").map_dict(
                {"English": "E", "British": "B", "Scottish": "S", "Welsh": "W"}
            ),
        )
        .with_columns(
            pl.when(pl.col("labels") == "B")
            .then(pl.col("scores").mean().over(["RGN21NM", "labels"]))
            .alias("sort_order")
        )
        .collect(streaming=True)
        .sort("sort_order", descending=True)
    )

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
            so.Range(linewidth=0.5, color="black"),
            so.Est(seed=0, errorbar="se"),
            so.Shift(y=0.2),
            # so.Dodge(),
        )
        .add(
            so.Text(offset=0, valign="center", color="black", fontsize=8),
            # so.Dot(edgewidth=1, edgecolor="black", artist_kws={"zorder": 10}),
            so.Agg(),
            # so.Dodge(),
        )
        .label(x="Confidence Value", y="")
        .theme({**axes_style("ticks")})
        .on(ax)
        .layout(engine="constrained")
        .plot()
    )
    fig.legends = []
    _.show()


if __name__ == "__main__":
    plt_zero_shot(Paths.PROCESSED / "places_zero_shot.parquet")
    plt.show()
