import pandas as pd
import polars as pl
from sentence_transformers import SentenceTransformer

from src.common.utils import MODEL, Paths

model = SentenceTransformer(MODEL)

places = pl.read_parquet(Paths.PROCESSED / "places.parquet")


def desc_tbl(places):
    select_query = (
        pl.col("masked").n_unique().alias("Total Comments"),
        pl.col("masked").str.split(" ").arr.explode().n_unique().alias("Unique Words"),
        pl.col("masked").str.split(" ").arr.explode().count().alias("Word Count"),
        pl.col("word").n_unique().alias("Total Places"),
    )

    totals = (
        places.select((pl.lit("Total").alias("RGN21NM"),) + select_query)
        .to_pandas()
        .style.format(thousands=",")
        .apply(lambda x: pd.Series("midrule:;", index=[0]), subset="RGN21NM")
        .highlight_between(subset=pd.IndexSlice[0, :], props="bfseries:;")
    )
    return (
        places.groupby("RGN21NM")
        .agg(select_query)
        .sort("Total Places", descending=True)
        .to_pandas()
        .style.format(thousands=",")
        .concat(totals)
        .hide(axis="index")
        # .to_latex(hrules=True)
    )


if __name__ == "__main__":
    print(desc_tbl(places))
