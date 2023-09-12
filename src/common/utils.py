import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

pl.Config.set_tbl_formatting("NOTHING")
pl.Config.with_columns_kwargs = True
pl.Config.set_tbl_dataframe_shape_below(True)
pl.Config.set_tbl_rows(6)

plt.rcParams.update({"font.size": 8, "text.usetex": False})


class Paths:
    RAW = Path(os.environ["DATA_DIR"])
    PROCESSED = Path("data/processed")


class Const:
    SEED = 42
    MODEL = "all-mpnet-base-v2"
    with open(Paths.PROCESSED / "exclude.txt", "r") as exclude:
        EXCLUDE = list({line.strip() for line in exclude.readlines()})

    CITIES = [
        "Glasgow City",
        "City of London",
        "City of Edinburgh",
    ]


def process_outs():
    places = pl.read_parquet(Paths.PROCESSED / "places.parquet")
    regions = gpd.read_parquet(Paths.RAW / "en_regions.parquet")
    lad = gpd.read_file(Paths.RAW / "lad-2021.gpkg")
    lad2rgn = pd.read_csv(Paths.RAW / "lad_to_region-2021.csv")[["LAD21CD", "RGN21NM"]]

    region_embeddings = regions.merge(
        pd.read_parquet(Paths.PROCESSED / "region_embeddings.parquet"), on="RGN21NM"
    )
    region_embeddings["embeddings"] = (
        region_embeddings["embeddings"].to_list()
        / np.linalg.norm(
            region_embeddings["embeddings"].to_list(), axis=1, keepdims=True
        )
    ).tolist()

    lad_embeddings = lad.merge(lad2rgn, on="LAD21CD", how="left").merge(
        pd.read_parquet(Paths.PROCESSED / "lad_embeddings.parquet"),
        left_on="LAD21NM",
        right_on="LAD22NM",
    )
    lad_embeddings["embeddings"] = (
        lad_embeddings["embeddings"].to_list()
        / np.linalg.norm(lad_embeddings["embeddings"].to_list(), axis=1, keepdims=True)
    ).tolist()
    lad_embeddings.loc[
        (lad_embeddings["RGN21NM"].isna())
        & (lad_embeddings["LAD21CD"].str.startswith("W")),
        "RGN21NM",
    ] = "Wales"
    lad_embeddings.loc[
        (lad_embeddings["RGN21NM"].isna())
        & (lad_embeddings["LAD21CD"].str.startswith("S")),
        "RGN21NM",
    ] = "Scotland"

    lad_embeddings["geometry"] = lad_embeddings["geometry"].simplify(1000)
    region_embeddings["geometry"] = region_embeddings["geometry"].simplify(1000)

    return places, regions, lad, region_embeddings, lad_embeddings
