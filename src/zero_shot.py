import polars as pl
from datasets import Dataset
from tqdm import tqdm
from transformers import pipeline

from src.common.utils import Paths

classifier = pipeline(
    "zero-shot-classification", model="typeform/distilbert-base-uncased-mnli", device=0
)
candidate_labels = ["British", "English", "Scottish", "Welsh"]

places = (
    pl.scan_parquet("./data/processed/places.parquet")
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
)
corpus = Dataset.from_pandas(places[["text"]].unique().to_pandas())

outs = list(tqdm(classifier(corpus["text"], candidate_labels)))

places = places.join(
    pl.DataFrame(outs).explode(columns=["labels", "scores"]),
    left_on="text",
    right_on="sequence",
)
places.write_parquet(Paths.PROCESSED / "places_zero_shot.parquet")
