import polars as pl
from datasets import Dataset
from tqdm import tqdm
from transformers import pipeline
from transformers.pipelines.base import KeyDataset

from src.common.utils import Paths


def zero_shot(places_path):
    # sourcery skip: for-append-to-extend, identity-comprehension, list-comprehension, simplify-generator
    classifier = pipeline(
        "zero-shot-classification",
        model="typeform/distilbert-base-uncased-mnli",
        device=0,
    )

    places = (
        pl.scan_parquet(places_path)
        .filter(pl.col("text").str.n_chars() > 10)  # fix empty strings
        .with_columns(
            pl.col("RGN21NM")
            .map_dict(
                {
                    "London": 1,
                    "South East": 1,
                    "East of England": 1,
                    "West Midlands": 2,
                    "Wales": 3,
                    "East Midlands": 3,
                    "South West": 3,
                    "North West": 3,
                    "Yorkshire and The Humber": 3,
                    "North East": 4,
                    "Scotland": 5,
                }
            )
            .apply(lambda x: str(x))
            .alias("cluster")
        )
        .with_columns(
            pl.col("cluster").map_dict(
                {
                    "1": "South East",
                    "2": "West Midlands",
                    "3": "England and Wales",
                    "4": "North East",
                    "5": "Scotland",
                }
            )
        )
        .collect()
        .sample(10_000)
    )

    corpus = Dataset.from_pandas(places[["text"]].unique().to_pandas())

    nationality_labels = ["British", "English", "Scottish", "Welsh"]
    nationality_outs = []
    for out in tqdm(
        classifier(KeyDataset(corpus, "text"), nationality_labels),
        total=len(corpus["text"]),
    ):
        nationality_outs.append(out)

    places = places.join(
        pl.DataFrame(nationality_outs).explode(columns=["labels", "scores"]),
        left_on="text",
        right_on="sequence",
    )


if __name__ == "__main__":
    places = zero_shot(Paths.PROCESSED / "places.parquet")
    places.write_parquet(Paths.PROCESSED / "places_zero_shot-tmp.parquet")
