import re
from pathlib import Path

import geopandas as gpd
import polars as pl
from tqdm import tqdm

from src.common.utils import Const, Paths


def join_to_regions(places_path):
    regions = gpd.read_file(Paths.RAW / "RGN_EN_BUC_2022.gpkg").drop(
        columns=["BNG_E", "BNG_N", "LONG", "LAT", "GlobalID"]
    )
    lad = gpd.read_file(Paths.RAW / "LAD_BUC_2022.gpkg").drop(
        columns=["BNG_E", "BNG_N", "LONG", "LAT", "GlobalID"]
    )
    df = (
        pl.read_parquet(places_path)
        .filter(pl.col("word").is_in(Const.EXCLUDE).not_())
        .select(
            [
                "idx",
                "h3_05",
                "text",
                "author",
                "word",
                "start_idx",
                "end_idx",
                "easting",
                "northing",
            ]
        )
        .to_pandas()
    )
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.easting, df.northing), crs=27700
    )
    lad = gpd.sjoin(lad, regions, how="left").drop(columns="index_right")
    gdf = gpd.sjoin(gdf, lad, how="left").drop(columns="index_right")

    gdf.loc[
        (gdf["RGN22NM"].isna()) & (gdf["LAD22CD"].str.startswith("W")),
        "RGN22NM",
    ] = "Wales"
    gdf.loc[
        (gdf["RGN22NM"].isna()) & (gdf["LAD22CD"].str.startswith("S")),
        "RGN22NM",
    ] = "Scotland"
    return pl.from_pandas(gdf.drop(columns="geometry"))


def read_places(places_path: Path, places_full: Path) -> pl.DataFrame:
    df = (
        pl.scan_parquet(places_path)
        .filter(pl.col("easting").is_not_null())
        .collect()
        .group_by(["word", "easting", "northing"])
        .map_groups(lambda x: x if len(x) < 5_000 else x.sample(5_000, seed=Const.SEED))
    )

    mask_df = (
        pl.scan_parquet(places_full)
        .filter(pl.col("idx").is_in(df["idx"].unique()))
        .collect()
    )

    text = mask_df[["idx", "text"]].unique()
    masks = mask_df[["idx", "start_idx", "end_idx"]]

    # NOTE: quite slow + sequential, maybe could be improved
    masked_text = []
    for row in tqdm(text.rows(named=True)):
        row_text = row["text"]
        text_mask = masks.filter(pl.col("idx") == row["idx"])
        for mask_row in text_mask.rows(named=True):
            word_len = mask_row["end_idx"] - mask_row["start_idx"]
            row_text = (
                f"{row_text[:mask_row['start_idx']]}"
                + "Ġ" * (word_len)
                + f"{row_text[mask_row['end_idx']:]}"
            )
        row_text = re.sub(r"Ġ+", "PLACE", row_text)
        masked_text.append(row_text)
    text = text.with_columns(masked=pl.Series(masked_text)).drop("text")

    return df.join(text, on="idx")


if __name__ == "__main__":
    places_path = Paths.RAW / "places-2023_04_11.parquet"

    place_regions = join_to_regions(places_path)
    place_regions.write_parquet(Paths.PROCESSED / "place_regions.parquet")

    places = read_places(
        Paths.PROCESSED / "place_regions.parquet",
        Paths.RAW / "places_full-2023_04_11.parquet",
    )

    places.write_parquet(Paths.PROCESSED / "places.parquet")
