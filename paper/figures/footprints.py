import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from paper.figures.place_vectors import process_embeddings

from src.common.utils import Paths

city_embeddings = pl.read_parquet(Paths.PROCESSED / "city_embeddings.parquet")
poi_embeddings = pl.read_parquet(Paths.PROCESSED / "poi_embeddings.parquet")
city_embeddings = (
    pl.concat([city_embeddings, poi_embeddings]).group_by("word").head(1000)
)

lad_embeddings = pl.read_parquet(Paths.PROCESSED / "lad_embeddings.parquet").filter(
    pl.col("LAD22NM").is_in(city_embeddings["LAD22NM"])
)
city = process_embeddings(city_embeddings.to_pandas())
lad_embeddings = process_embeddings(lad_embeddings.to_pandas())
city_mean = city.groupby(["word", "LAD22NM"]).mean(["vecx", "vecy"])

fig, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(
    data=city,
    x="vecx",
    y="vecy",
    hue="LAD22NM",
    edgecolor=None,
    s=5,
    legend=False,
    alpha=0.1,
    ax=ax,
)
plt.axis("off")
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(
    data=city_mean,
    x="vecx",
    y="vecy",
    hue="LAD22NM",
    edgecolor=None,
    s=50,
    legend=False,
    ax=ax,
)
for i, point in city_mean.reset_index().iterrows():
    ax.text(point["vecx"] + 0.005, point["vecy"] + 0.005, str(point["word"].title()))
plt.axis("off")
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(
    data=lad_embeddings,
    x="vecx",
    y="vecy",
    hue="LAD22NM",
    edgecolor=None,
    s=50,
    legend=False,
    ax=ax,
)
for i, point in lad_embeddings.reset_index().iterrows():
    ax.text(point["vecx"] + 0.005, point["vecy"] + 0.005, str(point["LAD22NM"].title()))
plt.axis("off")
plt.show()
