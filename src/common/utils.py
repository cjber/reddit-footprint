import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from tqdm import tqdm

pl.Config.set_tbl_formatting("NOTHING")
pl.Config.with_columns_kwargs = True
pl.Config.set_tbl_dataframe_shape_below(True)
pl.Config.set_tbl_rows(6)

plt.rcParams.update({"font.size": 8, "text.usetex": False})

SEED = 42
MODEL = "all-mpnet-base-v2"
CITIES = (
    pl.read_csv("data/cities.csv")
    .sort("2020", descending=True)["major town and city"]
    .str.to_lowercase()
    .unique()
    .to_list()
)
CITIES.extend(
    [
        "newcastle",
        "saint helens",
        "stockton",
        "hull",
        "stoke",
        "southend",
        "saint albans",
        "glasgow",
        "st. andrews",
    ]
)


class Paths:
    RAW = Path(os.environ["DATA_DIR"])
    PROCESSED = Path("data/processed")


with open(Paths.PROCESSED / "exclude.txt", "r") as exclude:
    EXCLUDE = list({line.strip() for line in exclude.readlines()})
