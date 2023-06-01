import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer

from src.common.utils import MODEL, SEED, Paths


def generate_embeddings(places):
    model = SentenceTransformer(MODEL)
    places = places.with_columns(pl.col("masked").str.to_lowercase())
    text = places["masked"].unique().to_list()
    embeddings = model.encode(text, show_progress_bar=True)

    return places.join(
        pl.DataFrame({"masked": text, "embeddings": embeddings}),
        on="masked",
    )


if __name__ == "__main__":
    places = pl.read_parquet(Paths.PROCESSED / "places.parquet")
    output = generate_embeddings(places)

    embeddings = []
    RGN21NM = []
    for group in output.partition_by("RGN21NM"):
        if len(group) > 1:
            embeddings.append(np.mean(group["embeddings"].to_numpy()))
        else:
            embeddings.append(group["embeddings"].to_numpy()[0])
        RGN21NM.append(group["RGN21NM"].unique().to_list()[0])

    embeddings = pl.DataFrame({"RGN21NM": RGN21NM, "embeddings": embeddings})

    embeddings.write_parquet(Paths.PROCESSED / "region_embeddings.parquet")
