import matplotlib.pyplot as plt
import polars as pl
import seaborn.objects as so
from seaborn import axes_style

from src.common.utils import EXCLUDE, Paths

zero_shot_path = Paths.PROCESSED / "places_zero_shot.parquet"


def plot_zero_shot(zero_shot_path):
    places = (
        pl.read_parquet(zero_shot_path)
        .filter((pl.col("word").is_in(EXCLUDE).is_not()))
        .unique(["text", "scores"])
        # some false locations
        .with_columns(
            RGN21NM=pl.when(
                pl.col("word").is_in(["aberdeen", "highlands", "st. andrews"])
            )
            .then("Scotland")
            .otherwise(pl.col("RGN21NM"))
        )
        .with_columns(
            pl.when(pl.col("labels") == "British")
            .then(pl.col("scores").mean().over(["RGN21NM", "labels"]))
            .alias("sort_order")
        )
        .sort("sort_order", descending=True)
        .with_columns(
            pl.col("labels").map_dict(
                {"English": "E", "British": "B", "Scottish": "S", "Welsh": "W"}
            ),
        )
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
            so.Text({"fontweight": "bold"}, offset=0),
            so.Agg(),
        )
        .add(
            so.Range(),
            so.Est(errorbar="ci", seed=0),
            so.Shift(y=0.3),
        )
        .scale()
        .label(x="", y="")
        .theme({**axes_style("ticks")})
        .on(ax)
        .layout(engine="constrained")
        .plot()
    )
    fig.legends = []
    _.show()


if __name__ == "__main__":
    plot_zero_shot(Paths.PROCESSED / "places_zero_shot-tmp.parquet")
