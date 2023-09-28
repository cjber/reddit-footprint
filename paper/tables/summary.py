import pandas as pd
import polars as pl

from src.common.utils import process_outs


def desc_tbl(places, lad_embeddings):
    select_query = (
        pl.col("text").n_unique().alias("Total Comments"),
        pl.col("text").str.split(" ").list.explode().n_unique().alias("Unique Words"),
        pl.col("text").str.split(" ").list.explode().count().alias("Word Count"),
        pl.col("word").n_unique().alias("Total Places"),
    )

    totals = (
        places.select((pl.lit("Total").alias("RGN21NM"),) + select_query)
        .to_pandas()
        .style.format(thousands=",", precision=2)
        .apply(lambda x: pd.Series("midrule:;", index=[0]), subset="RGN21NM")
        .highlight_between(subset=pd.IndexSlice[0, :], props="bfseries:;")
    )
    return (
        places.group_by("RGN21NM")
        .agg(select_query)
        .sort("Total Places", descending=True)
        .to_pandas()
        .style.format(thousands=",", precision=2)
        .concat(totals)
        .hide(axis="index")
        # .to_latex(hrules=True)
    )


if __name__ == "__main__":
    places, regions, lad, region_embeddings, lad_embeddings = process_outs()
    print(desc_tbl(places, lad_embeddings))
