import pandas as pd
import numpy as np
import polars as pl
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import scale

from src.common.utils import process_outs


def _emb_colour(v):
    if v < 0:
        color = "#ff5933"
    elif v > 0:
        color = "#33ff85"
    return f"color:{color};"


def desc_tbl(places, lad_embeddings):
    lad_embeddings = (
        pl.from_pandas(lad_embeddings.drop("geometry", axis=1))
        .with_columns(
            cosine_var=pl.col("embeddings")
            .map_elements(lambda x: np.array(x).mean(axis=0).std(axis=0))
            .over(pl.col("RGN21NM"))
        )
        .rename({"cosine_var": "Embeddings SD"})
        .unique("RGN21NM")
    )

    total = lad_embeddings["embeddings"].to_numpy().mean(axis=0).std(axis=0)
    lad_embeddings = lad_embeddings.with_columns(pl.col("Embeddings SD") - total)

    select_query = (
        pl.col("text").n_unique().alias("Total Comments"),
        pl.col("text").str.split(" ").list.explode().n_unique().alias("Unique Words"),
        pl.col("text").str.split(" ").list.explode().count().alias("Word Count"),
        pl.col("word").n_unique().alias("Total Places"),
    )

    totals = (
        places.select((pl.lit("Total").alias("RGN21NM"),) + select_query)
        .with_columns(std_emb=total - total)
        .rename({"std_emb": "Embeddings SD"})
        .to_pandas()
        .style.format(thousands=",", precision=2)
        .apply(lambda x: pd.Series("midrule:;", index=[0]), subset="RGN21NM")
        .highlight_between(subset=pd.IndexSlice[0, :], props="bfseries:;")
    )
    lad_embeddings = lad_embeddings.with_columns(
        pl.lit(scale(lad_embeddings["Embeddings SD"])).alias("Embeddings SD")
    )
    return (
        places.group_by("RGN21NM")
        .agg(select_query)
        .join(lad_embeddings[["RGN21NM", "Embeddings SD"]], on="RGN21NM")
        .sort("Total Places", descending=True)
        .to_pandas()
        .style.format(thousands=",", precision=2)
        .concat(totals)
        # .background_gradient(cmap="inferno", subset="Embeddings SD", vmin=-1, vmax=1)
        # .map(_emb_colour, subset="Embeddings SD")
        .hide(axis="index")
        # .to_latex(hrules=True)
    )


if __name__ == "__main__":
    places, regions, lad, region_embeddings, lad_embeddings = process_outs()
    print(desc_tbl(places, lad_embeddings))
