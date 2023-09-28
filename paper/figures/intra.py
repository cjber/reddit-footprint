import numpy as np
import polars as pl
from sklearn.preprocessing import scale

from src.common.utils import process_outs

_, _, _, region_embeddings, lad_embeddings = process_outs()


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


lad_embeddings = lad_embeddings.with_columns(
    pl.lit(scale(lad_embeddings["Embeddings SD"])).alias("Embeddings SD")
)[["RGN21NM", "Embeddings SD"]].to_pandas()

region_embeddings = region_embeddings.merge(lad_embeddings, on="RGN21NM")
region_embeddings.plot("Embeddings SD")
import matplotlib.pyplot as plt
plt.show()
