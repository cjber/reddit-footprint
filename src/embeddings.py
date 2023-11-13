import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer

from src.common.utils import Const, Paths


def generate_embeddings(places):
    model = SentenceTransformer(Const.MODEL)
    places = places.with_columns(pl.col("masked").str.to_lowercase())
    text = places["masked"].unique().to_list()
    embeddings = model.encode(text, show_progress_bar=True)

    return places.join(
        pl.DataFrame({"masked": text, "embeddings": embeddings}),
        on="masked",
    )


def mean_embeddings(output, by):
    embeddings = []
    grp_lst = []
    for group in output.partition_by(by):
        if len(group) > 1:
            embeddings.append(np.mean(group["embeddings"].to_numpy()))
        else:
            embeddings.append(group["embeddings"].to_numpy()[0])
        grp_lst.append(group[by].unique().to_list()[0])

    embeddings = pl.DataFrame({by: grp_lst, "embeddings": embeddings})
    return embeddings


if __name__ == "__main__":
    places = pl.read_parquet(Paths.PROCESSED / "places.parquet")
    output = generate_embeddings(places)

    out = mean_embeddings(output, "LAD22NM")
    out.write_parquet(Paths.PROCESSED / "lad_embeddings.parquet")

    out = mean_embeddings(output, "h3_05")
    out.write_parquet(Paths.PROCESSED / "h3_embeddings.parquet")

    out = mean_embeddings(output, "RGN22NM")
    out.write_parquet(Paths.PROCESSED / "region_embeddings.parquet")
