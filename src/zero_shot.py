import polars as pl
from datasets import Dataset
from tqdm import tqdm
from transformers import pipeline
from transformers.pipelines.base import KeyDataset

from src.common.utils import Paths


def zero_shot(places):
    classifier = pipeline(
        "zero-shot-classification",
        model="typeform/distilbert-base-uncased-mnli",
        device=0,
    )
    places = places.filter(pl.col("text").str.n_chars() > 10)  # fix empty strings
    # places = places.with_columns(
    #     pl.col("masked")
    #     .str.to_lowercase()
    #     .str.replace(
    #         r"england|english|wales|welsh|scotland|scottish|britain|british", "PLACE"
    #     )
    # )

    corpus = Dataset.from_pandas(places[["text"]].unique().to_pandas())

    nationality_labels = ["British", "English", "Scottish", "Welsh"]
    nationality_outs = list(
        tqdm(
            classifier(KeyDataset(corpus, "text"), nationality_labels),
            total=len(corpus["text"]),
        )
    )
    places = places.join(
        pl.DataFrame(nationality_outs).explode(columns=["labels", "scores"]),
        left_on="text",
        right_on="sequence",
    )
    return places


if __name__ == "__main__":
    places = pl.read_parquet(Paths.PROCESSED / "places.parquet")
    places = zero_shot(places)
    places.write_parquet(Paths.PROCESSED / "places_zero_shot.parquet")
