---
title: "Regional Identity and Cohesion Identified within the UK through Reddit Comments"
execute:
    echo: false
author:
  - name: Cillian Berragan
    affiliations:
      - name: University of Liverpool
        department: Geographic Data Science Lab
        # address: 1 Forestry Drive
        # city: Syracuse, NY
        # country: USA
        # postal-code: 13210
    orcid: 0000-0003-2402-304X
    email: c.berragan@liverpool.ac.uk
    # url: https://mm218.dev
format:
  arxiv-pdf:
    keep-md: true
    keep-tex: true
    linenumbers: true
    doublespacing: true
    runninghead: "A Preprint"
  # arxiv-html: default
keywords:
  - template
  - demo
bibliography: main.bib
filters:
    - abstract-section
project:
    execute-dir: $PWD
---


# Abstract

Both physical and online social interactions broadly decline in strength with geographic distance, and intra-community interaction strength often appears to correlate with pre-defined geographic administrative boundaries. These factors suggest that there are strong regional influences on social interactions, particularly given online social interactions are not directly restricted by distance. Regional identities between constituent countries within the UK are frequently studied from the context of structured surveys and census data, with language seen as a defining feature of these identities. In our work, we consider the exploration of semantic interaction strength, building a comparative analysis of corpora related to regions within the UK, determining shared semantic similarities between them. Building on this analysis, we establish the strength of regional Scottish, Welsh, English, and British identities within each region, using emergent properties of a large language model. We find that distinct groups emerge particularly in the South East and North East of England, while Scottish regional identity appears to be more distinct from England and Britishness compared with the Welsh identity.



# Introduction

Social interaction is typically studied in the context of mobility, using data sources like Census or transport records, where physical movement is restricted by distance and ease of connectivity between two locations [@rae2009;@titheridge2009]. In contrast to this, social interaction has also been studied using phone call data [@sobolevsky2013], and social media networks [@lengyel2015], where the spatial and temporal bounds of connectivity between two locations does not restrict interactions. Despite this however, many studies have found that geographic identities within communities still persist in these networks, with interaction strength influenced by both the geographic distance between them and their regional identity [@arthur2019;@ratti2010].

Social media also presents rich semantic information regarding locations through text associated with geotagged social media posts. Comparative analysis of corpora associated with geotagged locations similarly exhibit regionality [@yin2017a]; for example, tweets from the North East of England are statistically different compared with the South [@arthur2019].

In the context of political science, regional identities have been well studies through the use of structured survey or census data [@griffiths2022;@haesly2005;refs], which may provide explanation into the
inherent geographic restrictions that still appear with social media interactions. Language itself can be a strong indicator of regional identity [@haesly2005;refs], and past work has considered the regional variation in language that exists within social media networks [@huang2016;@goncalves2014;@perez2019].

Our paper explores the similarity of corpora with respect to locational mentions from data taken directly from text, without relying on geotagged metadata. This approach offers an alternative perspective for the analysis of social interaction, built directly from the semantic information associated with locations, rather than the location associated with social media users themselves. Collective semantic information from social media embeds the regional identity of locations across a continuous spectrum, allowing for the direct comparison between these identities and their relationships.

# Literature Review

Social interaction is a common theme in geographic research, typically focussing on the physical movement of individuals and populations to assume a level of inherent connectivity between distinct locations. These physical interactions are restricted by geographic distance [@miller2018;@patterson2015], following Tobler's first law of geography [@tobler1970] and the 'distance decay' effect [@taylor1983]. However, when examining administrative regions that exist within the United Kingdom, research has found that physical social interactions often correlated with these artificial boundaries [@ratti2010], suggesting that regional identities persist through these interactive networks, which disrupt the effect of distance.

Early research suggested that ease of communication regardless of distance meant that social ties were likely to become less geographically restricted [@cairncross1997], this assertion has however largely been found to be untrue. Many studies have found that despite the removal of this restriction in telecommunication, cohesive communities still persist, and interactions are broadly still affected by both geographic distance, and administrative boundaries [@onnela2011;@sobolevsky2013]. New forms of data provide social interactions through online social media networks like Twitter, which contribute geotagged interactions between users [@takhteyev2012;@arthur2019;@scellato2010]. These works similarly find that user interactions on social media are restricted by geographic distance and a 'distance decay' effect [@arthur2019;@stephens2015], and broadly conform with the arbitrary administrative boundaries that exist within geographic space [@arthur2019;@yin2017a].

The presence of these two geographic restrictions in social interactions in telecommunication and social media networks suggest that additional influencing factors are present.

## Regional identity

The concept of regional identities within populations has been well studied from the context of political research [@griffiths2022], particularly in relation to the shared national identity that becomes apparent within constituent countries in the United Kingdom [@haesly2005]. Methods for quantifying broad perspectives of regional identities have typically focussed on the use of surveys, questioning individuals beliefs that they belong to a particular nation.

When considering both Scottish and Welsh identities, @haesly2005 observe that in both cases, respondents display strong national pride and emphasise their distinction from England. The overall level of pride however was found to be stronger in Scotland, rooted in the richer history that creates a more distinct nation compared with Wales. Other studies have found that people from North East England exhibit a strong sense of regional identity, distinct from the rest of the country [@middleton2008;@nayak2016]. The strength of national identities may also be quantified through the use of Relative Territorial Identity (RTI) surveys, which asks respondents to rank their national identities on a scale ranging from 'not at all' to 'very strongly' [@griffiths2022]. @henderson2021 identify English nationalism by questioning a persons beliefs whether they are British or English. Similarly, a strong sense of Scottish identity can be observed by questioning whether a person feels more Scottish than British [@carman2014].

NOTE: Many works note the importance of language in defining a regional identity [@haesly2005;refs]...

NOTE: See [@huang2016] for linguistic variation stuff with Twitter data - link with rest. [@lee2007] - geographic text stuff, [@goncalves2014] - diatopic variation in geotagged micro blogs.

NOTE: [@perez2019] - regionalisms in social media posts.

NOTE: Slightly more detail about RTI; scottish identity stronger than welsh? British stronger than English?

NOTE: Follow with

Aims:

1. Calculate Semantic similarity measures between regional corpora
2. Generate cohesive meta-region clusters within the UK based on semantic information
3. Examine the distribution of regional identity across all UK regions

# Methodology

The following section gives an overview of our data source and the data processing methodology used in our paper. All code, analysis, and data are available on our [DagsHub repository](https://dagshub.com/cjber/reddit-geoext).

::: {.cell execution_count=2}

::: {.cell-output .cell-output-display}

[Reddit](https://reddit.com) is a public discussion, news aggregation social network, among the top 20 most visited websites in the United Kingdom. As of 2020, Reddit had around 430 million active monthly users, comparable to the number of Twitter users [@murphy2019;@statista2022]. Reddit is divided into separate independent _subreddits_ each with specific topics of discussion, where _users_ may submit _posts_ which each have dedicated nested conversation threads that users can add _comments_ to. Subreddits cover a wide range of topics, and in the interest of geography, they also act as forums for the discussion of local places. The [United Kingdom subreddit](https://reddit.com/r/unitedkingdom) acts as a general hub for related topics, notably including a list of smaller and more specific related subreddits. This list provides a 'Places' section, a collection of local British subreddits, ranging in scale from country level (`/r/England`), regional (`/r/thenorth`, `/r/Teeside`), to cities (`/r/Manchester`) and small towns (`/r/Alnwick`). In total there are 213 subreddits that relate to 'places' within the United Kingdom^[https://www.reddit.com/r/unitedkingdom/wiki/british_subreddits]. For each subreddit, every single historic comment was retrieved using the [Pushshift](https://pushshift.io/) Reddit archive [@baumgartner2020]. In total 8,282,331 comments were extracted, submitted by 490,535 unique users, between 2011-01-01 and 2022-04-17.

:::
:::


We extracted and geolocated all place names in this collection of comments using a custom-built geoparsing pipeline. To identify place names, we used a BERT transformer-based NER model trained on the WNUT 2017 dataset [@derczynski2017], available on the [HuggingFace Model Hub](https://huggingface.co/cjber/reddit-ner-place_names). We then implemented a disambiguation methodology using contextual place names and two gazetteers to geolocate place names; [OS Open Names](https://www.ordnancesurvey.co.uk/business-government/products/open-map-names) and 'natural' location types from the [Gazetteer of British Place Names](https://gazetteer.org.uk/). Processed comments consist of a collection of geolocated place names, alongside their natural language context sentence.

From this dataset, we retained any location mentioned by over 250 unique authors and with more than 500 context words. For each location we then took a random sample of 10,000 comments, ensuring each was represented equally, while also reducing the computational load of future processing.

## Similarity of Regional Corpora

Comparing the similarity between two or more distinct texts first relies on an appropriate method for processing the text into a numerical format. For each location we obtained a corpus of comments, consisting of sentences where each location is mentioned. These were then processed into a single vector, reflecting the semantic information attributed with locations. Typically, a TF-IDF approach is used to generate document embeddings [@daniel2007], however we found comparative analysis between embeddings did not always provide insightful information. Each vector shared similar properties, giving cosine similarities which did not result in any distinct variation between locations. This is likely a problem with the language between locations sharing similar properties, meaning the more nuanced semantic information is not captured through a TF-IDF method.

We therefore extracted embeddings from a deep neural network called a transformer. Unlike TF-IDF or simpler neural network models, transformers are able to use contextual information to generate word embeddings, meaning the same word in two different contexts will not share the exact same vector, capturing different embedded semantic information [@vaswani2017]. Additionally, transformers are _pre-trained_ on a large corpus of text, meaning general information regarding the English language is already embedded within the model, allowing for improved understanding of semantic information. These core features mean that embeddings generated from transformers are likely to capture information that allows for more the accurate comparative analysis. We generated embeddings using the `all-mpnet-base-v2` model from the `sentence-transformers` library in Python [@reimers2019]. Unlike a standard 'BERT'-like transformer, this library implements modifications to base models that more appropriately captures semantic information in their output embeddings.

Before calculating embeddings we first masked every mention of a location with a generic token 'PLACE', this ensured that when analysing embeddings, no explicit geographic information was captured accidentally. For example, London and the South East may mention matching locations frequently in each of their comments because they are geographically close. Once embeddings were generated for every comment in each regional corpus, the mean for each corpus was generated, giving a vector 768 decimal values for each UK region.

With a single vector for each region, we first calculated K-Means clusters to determine whether distinct groupings of regions could be identified across the UK. To visualise these clusters we used a PCA decomposition to reduce the dimensionality from 768 down to 2 dimensions. Finally, we calculated the cosine similarity between each and every location vector.

## Zero-Shot regional identities

Building on the concept of relative territorial identity measures, we construct a methodology to examine relative regional identity strength based on the semantic information embedded within comments. Semantic information is expected to capture both explicit information contributed by users; for example stating 'London is a British city', in addition to implicit semantic information that exists within language. For example the phrase 'bonnie Scotland' may suggest a strong regional identity due to the inclusion of Scottish slang^[See 'Scottish English' or 'Scots' [@stuart-smith2008].

To identify regional identities through semantic information, we build on the emergent properties of large language models, with enable a task known as 'Zero-Shot Classification'. This allows models to predict a class that was not seen during training, by generating a prompt that contains the labels required. For our task an example prompt may be constructed as follows:

```
Classify the following input text into one of the following four categories:
[British, English, Scottish, Welsh]

Input Text: I feel more Welsh than British.
```

The output would then be given as a sequence of confidence values for each label:

```
'labels': ['Welsh', 'British', 'English', 'Scottish']
'scores': [0.9902838468551636,
          0.007325825747102499,
          0.0013424316421151161,
          0.0010479396441951394]
```

# Results & Discussion

Table \ref{tbl-sum} gives an overview of the number of comments, word count and number of places that were identified within each administrative region of the UK. Our study concerns a subset of the full Reddit comment dataset, only considering place mentioned by more than 200 authors, leaving a total of XXX,XXX comments containing place names.

\begin{longtable}{lrrrr}
\caption{Summary of comments relating to each region in our study} \label{tbl-sum} \\
\toprule
RGN21NM & Total Comments & Unique Words & Word Count & Total Places \\
\midrule
\endfirsthead
\caption[]{Summary of comments relating to each region in our study} \\
\toprule
RGN21NM & Total Comments & Unique Words & Word Count & Total Places \\
\midrule
\endhead
\midrule
\multicolumn{5}{r}{Continued on next page} \\
\midrule
\endfoot
\bottomrule
\endlastfoot
London & 98,291 & 234,520 & 9,022,736 & 180 \\
Scotland & 71,855 & 212,745 & 7,260,246 & 153 \\
South West & 33,789 & 124,037 & 3,153,355 & 152 \\
South East & 38,484 & 140,807 & 3,450,759 & 140 \\
North West & 33,908 & 122,156 & 3,286,607 & 135 \\
Yorkshire and The Humber & 22,710 & 91,507 & 2,188,853 & 117 \\
East of England & 16,478 & 78,629 & 1,446,168 & 90 \\
West Midlands & 13,358 & 64,978 & 1,362,990 & 85 \\
East Midlands & 8,702 & 49,792 & 807,145 & 74 \\
Wales & 11,902 & 63,682 & 1,290,962 & 59 \\
North East & 9,879 & 59,139 & 1,057,767 & 59 \\
\midrule \bfseries Total & \bfseries 341,189 & \bfseries 610,164 & \bfseries 34,327,588 & \bfseries 531 \\
\end{longtable}



::: {.cell execution_count=4}

::: {.cell-output .cell-output-display}
![Average transformer vector associated with each location corpus coloured by K Means clusters where $K=5$. (A) PCA decomposed into 2 dimensions. (B) Geographic location of clusters.](main_files/figure-pdf/fig-pca-output-1.pdf){#fig-pca}
:::
:::


@fig-pca gives K Means clusters for transformer embeddings decomposed into two dimensions with $k=5$. These Clusters indicate regional corpora that share similar semantic properties. it is worth noting however, that while points that are closer together likely indicate increased similarity, the position of these points reflect PCA decomposed values, which capture less information compared with the clusters calculated on non-decomposed vectors. Notably Scotland, the North East, and West Midlands each appear as single values in unique clusters, suggesting semantically distinct corpora from the rest of the country. There are two further clusters which encompass the South East of England (Cluster 1), and the rest of England and Wales (Cluster 3). @fig-pca (B) reveals that clusters do broadly appear to reflect distance-restricted geographic properties, while also capturing some divergences from this where neighbouring administrative regions appear in distinct clusters.

::: {.cell execution_count=5}

::: {.cell-output .cell-output-display}
![Cosine similarity between each and every location related transformer vector embedding. Green highlights indicate the highest value in each row, while red indicates the lowest value in each row.](main_files/figure-pdf/fig-similarity-output-1.pdf){#fig-similarity}
:::
:::


With our high dimensional transformer embeddings we compare the cosine similarity between them on @fig-similarity. The highest and lowest similarity score for each location is highlighted in red and green respectively. As with @fig-pca, corpora in Scottish cities appear to largely share similarities, with Glasgow and Edinburgh sharing their highest similarity values. The city with the lowest similarity to the most other locations is Oxford, which shares low values with cities in Scotland, as well as Liverpool and Manchester. London again stands out, with overall very low similarities with all other cities, but the highest similarity with Manchester.

## Identities

@fig-boxplot shows the distribution of regional identities identified through our zero-shot classification methodology.

::: {.cell execution_count=6}

::: {.cell-output .cell-output-display}
![Zero Shot classification of each corpus into regional identities; [B]ritish, [E]nglish, [S]cottish, [W]elsh. Values show mean confidence value across each comment. Descending order by [B]ritish confidence.](main_files/figure-pdf/fig-boxplot-output-1.pdf){#fig-boxplot}
:::
:::


The model makes some interesting choices depending on the sentiment of the input text. For example if we input "I love Scotland!" the model labels the text as Scottish with 94% confidence, however if we change the text to "I hate Scotland!" the model labels the text as British with 60% confidence, suggesting that the model understands that personal national sentiment is typically more likely to be positive.

Unusual patterns emerge if we select particular cities; hating Aberdeen is more likely to be Scottish, while hating Edinburgh is more likely to be British. Hating 'the North' is more likely to be British, while hating 'the South' is more likely to the Scottish. Most interestingly, the model appears mostly reluctant to label any text explicitly talking about English locations as English, and prefers to use British. This perhaps reflects the overall general national identity that is associated with England as a country within the UK, compared with Scotland and Wales.

# Conclusion

Our paper demonstrates the ability to compare Reddit comments relating to cities across the UK, using document embeddings generated from a transformer neural network. Instead of focussing on physical interactions between people or social media interactions, our work identifies relationships between cities through their semantic footprint, and analysing each corpus computationally allows for direct comparisons between cities through clustering and cosine similarity.

Our analysis reveals distinct clusters which largely reflect geographic proximity of locations, however, interesting deviations from proximity do emerge. Oxford and Cambridge are both clustered and share a high cosine similarity, but generate the lowest similarity with many other locations in the UK, including London. London in particular appears distinct from the rest of the UK, while cities that are not geographically close exhibit clustering and high similarity, such as Liverpool and Newcastle.

The information generated through our work presents an alternative view of relationships between cities that are not captured by existing data sources, all of which rely on explicit geographic coordinate information. Instead, we build similarities and clusters directly from the semantic information that exists within their respective corpora. Unlike traditional data, which captures objective social interactions between regions, the deviations from the restriction of geographic distance between several cities in our work appears to reflect the more subjective language that shapes the cultural and perceived identity of regions, and the relationships between them.

While our work enables the direct numerical comparison between city-based corpora, it cannot explain the similarities and dissimilarities between them. Additional work may explore the use of topic-modelling to identify shared topics between locations, and differences in the sentiment towards these topics may explain dissimilarity.

NOTES: Our results appear to broadly conform with general studies using both mobility data, survey, and geotagged social media which suggest there is a strong regional identity with Scotland, but less so with Wales. We additionally identify regions within the UK that also appear to have some separation, like the North East, and quantitatively identify regions with the strongest overall connection with the rest of the country; the North West. See 'More Scottish than British' [@carman2014].

# References {-}
