import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
import polars as pl
from tqdm import tqdm

from src.common.utils import EXCLUDE, SEED, Paths


def join_to_regions(places_path, england_path, scotland_path, wales_path):
    wales = gpd.GeoDataFrame(
        {"geometry": [gpd.read_file(wales_path).unary_union]},
        crs=27700,
    )
    wales["RGN21NM"] = "Wales"
    scotland = gpd.GeoDataFrame(
        {"geometry": [gpd.read_file(scotland_path).unary_union]},
        crs=27700,
    )
    scotland["RGN21NM"] = "Scotland"

    regions = pd.concat(
        [
            gpd.read_file(england_path)[["RGN21NM", "geometry"]],
            wales,
            scotland,
        ]
    )
    regions["geometry"] = regions.simplify(1000)
    regions.to_parquet(Paths.PROCESSED / "en_regions.parquet")

    df = pl.read_parquet(places_path).to_pandas()

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["easting"], df["northing"]), crs=27700
    )

    regions = gpd.sjoin(gdf.drop_duplicates(subset="geometry"), regions)

    return pl.from_pandas(regions[["easting", "northing", "RGN21NM"]]).join(
        pl.from_pandas(
            df[
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
            ]
        ),
        on=["easting", "northing"],
    )


def read_places(places: Path, places_full: Path) -> pl.DataFrame:
    df = (
        pl.scan_parquet(places)
        .unique()
        .filter(
            (pl.col("easting").is_not_null()) & (pl.col("word").is_in(EXCLUDE).is_not())
        )
        .with_columns(
            pl.count().over("word").alias("word_count"),
            pl.col("author")
            .n_unique()
            .over(["word", "easting", "northing"])
            .alias("author_count"),
        )
        .filter((pl.col("author_count") > 250))
        .sort("author_count", descending=True)
        .collect()
        .groupby(["word", "easting", "northing"])
        .apply(lambda x: x if len(x) < 5_000 else x.sample(5_000, seed=SEED))
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
        row_text = re.sub(r"Ġ+", "", row_text)
        masked_text.append(row_text)
    text = text.with_columns(masked=pl.Series(masked_text)).drop("text")

    return df.join(text, on="idx")


if __name__ == "__main__":
    places_path = Paths.RAW / "places-2023_04_11.parquet"
    wales_path = Paths.RAW / "wales_bdry.gpkg"
    scotland_path = Paths.RAW / "scot_bdry.gpkg"
    england_path = Paths.RAW / "en_regions.gpkg"

    place_regions = join_to_regions(
        places_path, england_path, scotland_path, wales_path
    )
    place_regions.write_parquet(Paths.PROCESSED / "place_regions.parquet")

    places = read_places(
        Paths.PROCESSED / "place_regions.parquet",
        Paths.RAW / "places_full-2023_04_11.parquet",
    )
    places.write_parquet(Paths.PROCESSED / "places.parquet")
