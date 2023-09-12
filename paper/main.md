---
title: "Mapping Semantic Regional Variation in Great Britain through Reddit Comments"
execute:
    echo: false
    warning: false
    cache: true
author:
  - name: Cillian Berragan
    affiliations:
      - name: University of Liverpool
        department: Geographic Data Science Lab
    orcid: 0000-0003-2198-2245
    email: c.berragan@liverpool.ac.uk
format:
  arxiv-pdf:
    keep-md: true
    linenumbers: true
    doublespacing: true
keywords:
  - semantics
  - social media
  - natural language processing
bibliography: /home/cjber/drive/bib/zbib.bib
csl: https://www.zotero.org/styles/harvard-cite-them-right
filters:
    - abstract-section
---


# Abstract

Observed regional variation in geotagged social media text is often attributed to dialects, where features in language are assumed to exhibit region-specific properties. While dialects are seen as a key component in defining the identity of regions, there are a multitude of other geographic properties that may be captured within natural language text. In our work, we consider locational mentions that are directly embedded within comments on the social media website Reddit, providing a range of associated semantic information, and enabling deeper representations between locations to be captured. Using a large corpus of Reddit comments from UK related local discussion subreddits, we identify place names using a transformer-based named entity recognition model. Embedded semantic information is then generated from these comments and aggregated into local authority districts, representing the semantic footprint of these regions. These footprints broadly exhibit spatial autocorrelation, with clusters that conform with administrative borders like Wales, Scotland, and London. Wales, Scotland, and London demonstrate notably different semantic footprints compared with the rest of the UK, which may be explainable through the perception of national identity associated with these regions.



# Introduction

While formal geographic regions within Great Britain are typically designed for administrative and political purposes, they are non-restrictive in how populations can move between them. The level of geographic cohesion between regions across the UK is often studied from the context of mobility, where data sources like Census or transport records describe the physical movement of populations and individuals across geographic space [@rae2009;@titheridge2009], or through non-physical networks using phone records [@sobolevsky2013;@reades2009;@zheng2015;@lambiotte2008], and social media [@lengyel2015;@arthur2019;@sui2011]. When these networks are examined, cohesive regions develop, which broadly appear to correlate with administrative boundaries [@arthur2019;@ratti2010].

As with social interactions, language is also known to exhibit regional variation [@trudgill2004], and certain dialects contribute to the concept of identity in regions across the UK [@haesly2005;@llamas2014;@llamas2009]. While variations in dialects may be explored from the perspective of survey data [@clopper2016], alternative approaches have considered statistical methods for determining lexical variation in geographically bounded social media networks [@russ2012;@doyle2014;@huang2016;@goncalves2014;@perez2019;@arthur2019;@eisenstein2014]. Comparative analysis of corpora associated with geotagged locations have similarly been shown to exhibit regional properties; for example, Tweets originating in the North East of England are noticeably different compared with the South [@arthur2019].

While social media text has been shown to exhibit regional variation, these works prevalently rely on the use of geotagged posts. The content of geotagged posts is often unrelated to the geotagged location, meaning any geographically associated lexical variation is often attributed to dialects, despite only presenting a single component of language that may exhibit geographic heterogeneity. Instead, our paper explores the similarity of corpora with respect to locational mentions taken from social media text, embedded directly within a related semantic context. Place names may be attributed with coordinate information through a process called geoparsing, where names are identified automatically in text through a natural language processing technique called named entity recognition, and associated an entry in a gazetteer. This approach offers an alternative perspective for the analysis of geographic identity, built directly from the semantic information associated with locations, rather than the location associated with social media users themselves. Semantic information therefore represents a broad range of topics and associations directly relating to geographic locations, rather than just the dialect of contributing users.

Our hypothesis is that related semantic information regarding locations from social media embeds a regional footprint, representing the collective geographic knowledge of each individual user in our corpus. Any geographic knowledge extracted from social media represents an informal, place-based understanding of geography, that more appropriately reflects a cognitive understanding of associations between locations [@sui2011;@berragan]. Footprints are represented through embeddings; numerical representations of text that capture semantic information, allowing for a direct comparison between them. To explore why certain regions in the UK appear more semantically isolated, we consider the link between identity, querying a large language model to determine the perception of the strength of national identities in UK regions.

@sec-literature first gives an overview of work exploring semantic variation in social media text, place-based geographic knowledge, and regional identities. @sec-methodology first describes our data, then outlines the processing used to generate regional embeddings for comparison and our method for identifying regional identities within these comments. @sec-results presents our results and @sec-conclusion concludes with suggestions for future work.

We have identified the following key aims that are addressed in this paper:

> **Aim 1:** Identify cohesive meta-region clusters within Great Britain based on semantic information

> **Aim 2:** Calculate Semantic similarity measures between regional corpora

> **Aim 3:** Examine the distribution of perceived regional identity across all British regions

# Geographic Variation in Social Media Language {#sec-literature}

<!-- NOTE: This really relates more to the idea of dialects, which we do not cover.

When the concept of regional identities have been studied in the context of political science [@griffiths2022], language is often considered to be a key component [@haesly2005;@llamas2014;@llamas2009]. The Scottish identity, for instance, is strongly associated with a sub-dialect of English known as 'Scottish English' which includes words and phrases that are uncommon outside of Scotland [@stuart-smith2008]. Similarly, language in the North East of England is considered a defining feature of its regional identity, despite having no formal identity separate to England [@nayak2016]. Survey methods for quantifying broad perspectives of such regional identities have typically focussed on questioning individuals' beliefs that they belong to a particular nation. For example the Regional Territorial Identity (RTI) method may rank the belief that a person is 'British', or 'Scottish' on a scale from 'not at all' (0) to 'very strongly' (10) [@griffiths2022].

NOTE: Find literature on 'londoners', 'northeners', 'cornish' [@daecon2007] etc.
 -->

While it is well understood that, despite the advancement of communication technologies, social ties still appear to be geographically restricted [@onnela2011;@sobolevsky2013;@ratti2010;@scellato2010;@takhteyev2012], it is less well understood why these geographic restrictions appear to correlate with administrative boundaries [@ratti2010;@arthur2019;@stephens2015;@yin2017a]. When considering the lexical and semantic properties of text associated with geotagged social media data, geographically cohesive groupings occur [@huang2016;@eisenstein2014;@goncalves2014;@goncalves2014;@arthur2019@arthur2019], which conforms with the idea of dialects are an important component in the identity of regions [@haesly2005;@llamas2014;@llamas2009]. Dialects however, only present a single component of language that may be associated with geographic locations.

The prevalence of social media data for use in geographic research has generated a renewed interest in the concept of 'place' [@wagner2020;@purves2019;@westerholt2018a;@berragan], as contributions to social media are theorised to contribute informal knowledge that represents a 'place-based' understanding of geography [@goodchild2011]. In the context of language, this place-based knowledge is contributed through 'vernacular geography', which describes the natural language used when informally describing geographic locations [@gao2017a;@goodchild2011;@waters2003;@hollenstein2008]. This informal knowledge incorporates associated biases regarding locations, better representing human associations between locations compared with formal administrative aggregations. In this sense, associations of geography drawn from social media capture place through a 'bottom-up' approach, building knowledge through experience rather than administrative formalisations [@agnew2005;@sui2011]. While many works consider the formalisation of place through social media data, few have considered how the semantic properties of text may reveal semantic associations between regions, captured through vernacular geography.

## Measuring Regional Variation in Language

Regional variations in linguistic characteristics are often quantified from a statistical analysis of lexicons, considering that regions exhibit subtle differences in dialects that may be captured. For example, studies have considered regional differences in common words on Twitter [@huang2016;@eisenstein2014;@goncalves2014;@arthur2019;@russ2012;@han2012;@doyle2014;@zheng2018], or identified spoken differences in dialects from active participation methods [@clopper2016]. While few explicitly quantify inter-regional cohesion through language, cosine similarity measures have been used to examine geotagged lexicons from Twitter within England and Wales [@arthur2019]. Results demonstrated that the Welsh lexicon appeared to be the most dissimilar to all other regions, while Northern English regions were more similar to each other compared with Southern regions. Alternatively vague 'vernacular' regions have been defined through an analysis of the dialects, where geotagged lexical alternations in Tweets were used to establish a clear North and South divide in the USA [@huang2016].

Given dialects are considered key in defining regional identities, these observations likely in part explain the coherent regions that form through both physical and online social interactions. Users are more likely to interact with others that share a similar identity, which is in part definable through the semantic properties of language. Despite this, there are a multitude of other properties that contribute to the perceived identity of geographic regions. This limitation primarily stems from the ability to capture geographic properties from existing data sources, where the content of Tweets is often unrelated to the geotagged location. In these works, the only geographic property that is captured in geotagged posts, is the dialect of the contributing author, with no method for determining any semantic information that directly relates to geography.

With the ability to accurately geoparse text, locations can instead be extracted that are embedded directly within a related semantic context, without relying on geotags. This method enables semantic information in text to be directly associated with defined geographic locations, providing a context covering a broad range of topics relating to locations, built from the mental associations of contributing authors. Given social media contributes informal geographic knowledge [@sui2011], observations generated from this data conforms with a general understanding of 'place', incorporating cultural information, locational characteristics, and capturing cognitive biases. 

In this work we aim to generate a new measure of cohesion between regions through an examination of text associated with locations, extracted from comments on the social media website Reddit. While past work has examined cohesion within regions from the perspective of social media networks, or by examining lexicons associated with geotagged social media messages, we examine regional variations derived from text associated with directly embedded place names. Unlike using geotags, which ascribe linguistic features such as dialect to specific locations, our method instead captures any comment than mentions a location alongside its context. Quantified information therefore does not reflect dialects associated with locations, but common semantic associations, embedding cultural information, or location specific topics and opinions. Locations with higher cohesion with the rest of Great Britain are likely to reflect more similar semantic properties, which more appropriately reflect the broader concept of a regional identity.

## Embedding Semantic Footprints

Comparative analysis of two or more distinct texts first relies on an appropriate method for processing the text into a numerical format. Typically, a TF-IDF approach is used to generate document embeddings [@daniel2007], which assigns word importance based on the frequency of mentions within a corpus. TF-IDF however does not have the capability for capturing broader semantic information, given there is no knowledge of the meaning behind words. Large Language Models (LLMs) instead are pre-trained on a very large corpus of natural language text, which, alongside their architecture, enables them to more appropriately consider semantic information [@devlin2019]. Similarly to TF-IDF, text is input into these models and output as a numerical representation, which embeds words as high dimensional vectors, capturing contextual semantic information. 

This approach differs from past work that only considered a lexical analysis, where semantic information and context is not preserved, instead building vectors that act as representations of locations identified in our corpus, which we name 'semantic footprints'. Given semantic information is preserved, our results reflect the deeper associations between geographic locations, built from a multitude of contexts and perspectives, forming an aggregate representation. Any geographically cohesive footprints therefore demonstrate a direct association between geography and language, which hasn't been captured previously.

<!---->
<!-- NOTE: These language models embed contextual semantic information within their model weights, represented as high dimensional token embeddings. Unlike previous methods for place name identification, these models have several benefits; -->
<!---->
<!-- * **Contextual word embeddings**: Words in different contexts have different meanings (e.g. Bath the place is different to a bath) -->
<!---->
<!-- * **Pre-trained**: Existing semantic information regarding the English language is embedded within the model during pre-training, which increases model accuracy during training -->
<!---->
<!-- * **Attention**: Context is preserved, which improves the semantic understanding of text -->

## Zero Shot Classification

As LLMs are pre-trained on a large corpus of natural language text, building representations of this text that emulate a human understanding of language. Language models therefore in theory represent the collective of humans that contributed the natural language text they were trained on. In addition to factual information, when posed with non-deterministic questioning, these models are able to contribute the biased information that is incorporated into their model weights.

NOTE: Here write up summary of paper.

# Methodology {#sec-methodology}

::: {.cell execution_count=2}

::: {.cell-output .cell-output-display}

[Reddit](https://reddit.com) is a public discussion, news aggregation social network, and among the top 20 most visited websites in the United Kingdom. In 2020, Reddit had around 430 million active monthly users, comparable to the number of Twitter users [@murphy2019;@statista2022]. Reddit is divided into separate independent _subreddits_ each with specific topics of discussion, where _users_ may submit _posts_ which each have dedicated nested conversation threads that users can add _comments_ to. Subreddits cover a wide range of topics, and in the interest of geography, they also act as forums for the discussion of local places. The [United Kingdom subreddit](https://reddit.com/r/unitedkingdom) acts as a general hub for related topics, notably including a list of smaller and more specific related subreddits. This list provides a 'Places' section, a collection of local British subreddits, ranging in scale from country (`/r/England`), region (`/r/thenorth`, `/r/Teeside`), to cities (`/r/Manchester`) and small towns (`/r/Alnwick`). In total there are 213 subreddits that relate to 'places' within the United Kingdom^[https://www.reddit.com/r/unitedkingdom/wiki/british_subreddits]. io/) Reddit archive [@baumgartner2020] by @berragan. we use the corpus generated by @berragan, which consists of a collection of all Reddit comments taken from each UK related subreddit, with place names identified by a custom transformer-based named entity recognition model^[https://huggingface.co/cjber/reddit-ner-place_names]. In total 8,282,331 comments were extracted, submitted by 490,535 unique users, between 2011-01-01 and 2022-04-17. \ref{tbl-example} gives an example entry from this geoparsed Reddit corpus.

In total our corpus consisted of 52,169 unique locations, with a highly skewed distribution in mentions. Most locations were only mentioned a single time, while 'London' was mentioned in almost 300,000 comments. To reduce this skew, sampled any location mentioned more than 5,000 times, retaining only up to 5,000 randomly sampled comments. The goal with this processing was to ensure that our generated embeddings did not simply become biased towards the word embedding for a single location, and instead capture a broader sense of an aggregate administrative region.

:::
:::


\begin{table}[tb]
\centering
\caption{Summary of comments relating to each region in our study}
\label{tbl-example}
\begin{tabular}{lll}
\toprule
Variable & Value & Description \\
\midrule
text & A Mexicana meal with extra wings  & Comment \\
 & from Tex in Leytonstone. &  \\
word & leytonstone & Identified Place Name \\
easting & 539,268 & Place Name Easting \\
northing & 187,540 & Place Name Northing \\
region & London & Administrative Region \\
lad & Waltham Forest & Local Authority District \\
author & t2\_eklyq & Anonymised Unique Author ID \\
word\_count & 855 & Total location mentions \\
author\_count & 431 & Unique authors mentioning this location \\
\bottomrule
\end{tabular}
\end{table}



## Generating Embeddings

To determine whether language relating to geographic locations in our Reddit corpus exhibits regional cohesion, we first generate semantic embeddings for each comment in which a location was mentioned. Embeddings were generated using the `sentence-transformers` Python library [@reimers2019], using the `all-mpnet-base-v2` model. With our selected embedding model, we performed the following steps to generate embeddings for each Local Authority District (LAD) in Great Britain.

1. Masked any place name with a generic token: 'PLACE'
2. Generate sentence embeddings for each comment
3. Group embeddings by LAD using identified locations, taking the mean embedding

To visualise the outputs from this processing we consider an example comment $s_1 = \text{"I live in London."}$, shown on @eq-dims. 

$$
\begin{aligned}
\mathit{s_{1}} &= \text{'I live in \textit{London}'} \\
\textbf{1. }\downarrow \\
\mathit{s_{1}} &= \text{'I live in \texttt{PLACE}'},
\end{aligned}
\qquad
\begin{aligned}
\textbf{2. }\mathit{s_{i}} \rightarrow 
\begin{bmatrix}
x_{1} \\
x_{2} \\
\vdots\\
x_{n}
\end{bmatrix},
\end{aligned}
\qquad
\begin{aligned}
\textbf{3. }\mathit{F_{j}} = 
\begin{bmatrix}
x_{1,1} & x_{1,2} & \cdots & x_{1,t} \\
x_{2,1} & x_{2,2} & \cdots & x_{2,t} \\
\vdots  & \vdots  & \ddots & \vdots  \\
x_{n,1} & x_{n,2} & \cdots & x_{n,t}
 \end{bmatrix} \rightarrow \begin{bmatrix}
\bar{x_{1}} \\
\bar{x_{2}} \\
\vdots \\
\bar{x_{n}}
\end{bmatrix}
\end{aligned}
$${#eq-dims}

In @eq-dims, $n$ is the `sentence-transformers` embedding dimension (768), and $t$ is the total number of unique comments that relate to a single LAD region $j$. $x_i$ values are in (2) are model weights that represent the embedding for the comment $s_i$, capturing semantic information. All comment embeddings associated with a LAD $F_j$ are then processed into one dimension by taking the mean (3).

By masking place names we ensure that no comment embeddings accidentally incorporated geographically grounded information. For example, comments in South Eastern local authorities are likely to frequently mention London, given they are geographically close. Embeddings for these locations would therefore capture an association through the mention of London, rather than general semantic information. For our work, we want to exclude any geographic information, ensuring that embeddings solely capture semantic associations.

Given that transformers are a fairly new architecture in natural language processing, and the creation of these models require significant computational resources and training time, their use to date has been limited in related research. Our choice to use the transformer architecture stems from the emphasis we place on the extraction of nuanced and contextual semantic information, which is lost with more statistical methods. It should be noted however that while TF-IDF lexical methods are less complex, they are typically more interpretable; for instance, words that contribute importance to an embedding may be extracted from a TF-IDF model. The numerical representations of any text generated by transformers are not directly interpretable in this manner.

## Spatial Clustering and Autocorrelation

To explore Aim 1 we use Agglomerative clustering to generate hierarchical clusters for our LAD embeddings. Agglomerative clustering is able to automatically select the optimal number of clusters based on a distance threshold, which in our data was determined to be 3. Clusters were visualised both geographically, and as a scatter plot, using a UMAP decomposition to reduce the dimensionality down to 2 dimensions. Additionally, we visualise the proportion of clusters presented within each administrative region in Great Britain.

NOTE: Update the description of visualisations, e.g. using UMAP now, and explain more deeply how they work.
NOTE: The autocorrelation we now explore are 2 dimensional, which have a low cosine similarity and therefore likely have different explanatory components. Mention this here, using UMAP with 2x morans I and two LISA analysis, to determine whether there is any patterns.

We additionally identify the global level of spatial autocorrelation through the Moran's I metric, and identify geographic clusters of shared semantic attributes, using a Local Indicators of Spatial Autocorrelation (LISA) analysis [@anselin1995;@rey2023]. As spatial autocorrelation analysis is only univariate, we consider the two UMAP decomposed dimensions for this analysis, rather than the high dimensional LAD embeddings. Given the numerical value of this decomposed embedding does not convey any definable information based on its magnitude, we do not distinguish between HH or LL instances in our LISA analysis. These represent both those of low values near other low values, and high values near other high values, meaning all LAD embeddings with high spatial autocorrelation.

To explore intra-community cohesion, we calculate cosine similarities between LADs within each administrative region of the UK, and take the standard deviation. These are then subtracted from the global standard deviation to give a relative measure, all values are scaled to have a mean of zero and standard deviation of one.

### Semantic Similarity

To explore Aim 2, we determine the semantic similarity between regions across Great Britain, by grouping embeddings into their respective administrative region, taking the mean embedding value across each region. We then calculate the cosine similarity between each and every embedding, demonstrating the inter-region cohesion across Great Britain.

Cosine similarity is a common metric for comparing embeddings, as it is invariant to the magnitude of the vectors, and only considers the direction. This is important as the magnitude of embeddings is not meaningful, and only the direction of the vector conveys information. For example, the embedding for 'London' is not twice as important as the embedding for 'Manchester', and therefore the magnitude of the embedding is not meaningful.

## Zero-Shot regional identities

To explore Aim 3, we build on the concept of relative territorial identity measures, and construct a methodology to examine relative regional identity strength based on the semantic information embedded within comments. Semantic information is expected to capture both explicit information contributed by users; for example stating 'London is a British city', in addition to implicit semantic information that exists within language. For example the phrase 'bonnie Scotland' may suggest a strong regional identity due to the inclusion of Scottish slang^[See 'Scottish English' or 'Scots'; [@stuart-smith2008]].

To identify regional identities through semantic information, we build on the emergent properties of large language models, which enable a task known as 'Zero-Shot Classification'. This allows models to predict a class that was not seen during training, by generating a prompt that contains the labels required. For our task an example prompt may be constructed as follows:

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

# Results & Discussion {#sec-results}

::: {.cell execution_count=4}

::: {.cell-output .cell-output-display}

We first note on the data quality and bias used in our paper, a common concern in large social media corpora. @berragan note that 1% of all users contribute 32% of all identified place names, which represents the top 2,079 users in the full corpus. In our subset, we similarly find that 1% of users (1,698) mention 29% of our place names.

:::
:::


As described in our methodology, we found that transformer embeddings provide more usable information for comparison between regions compared with a TF-IDF approach. This suggests that we have likely captured an increased contribution of semantic information compared with past work that has considered only using lexicon based approaches. Additionally, it is worth mentioning that averaging down our embeddings does likely remove a significant proportion of semantic information, however, averaging embeddings has been demonstrated to perform well for many transformer-embedding based tasks. For example, sentence embeddings are simply the average of word embeddings, and are used in many NLP tasks like topic identification or sentence similarity [@reimers2019].

Unlike past work that has used count-based lexicons to generate corpus similarity measures between regional social media posts [e.g. @arthur2019], we instead generate transformer embeddings from our comments. This likely increases the depth of semantic information captured within these embeddings, especially considering informal region specific language is likely highly context dependent.

::: {.cell execution_count=5}

::: {.cell-output .cell-output-display}

Table \ref{tbl-sum} gives an overview of the number of comments, word count and number of places that were identified within each administrative region of the UK. Our study concerns a subset of the full Reddit comment dataset, subsetting each location to a maximum of 5,000 mentions, leaving a total of 830,770 comments containing place names. Comments range from 1 to 3,555 words in length, with a mean length of 79. On Table \ref{tbl-sum} the 'Embeddings SD' values give an approximate indication of the intra-community cohesion, lower values indicting stronger cohesion compared with the global average, meaning a higher proportion of shared semantic information.

:::
:::


\begin{table}[tb]
\centering
\caption{Summary of comments relating to each region in our study}
\label{tbl-sum}
\begin{tabular}{lrrrrr}
\toprule
RGN21NM & Total Comments & Unique Words & Word Count & Total Places & Embeddings SD \\
\midrule
Scotland & 181,831 & 437,746 & 23,218,279 & 8,052 & 1.24 \\
South East & 107,134 & 308,491 & 11,849,441 & 5,679 & -0.89 \\
London & 206,280 & 422,036 & 23,868,430 & 5,164 & 2.14 \\
South West & 85,960 & 267,388 & 9,680,548 & 5,090 & 0.55 \\
North West & 88,789 & 259,154 & 10,650,018 & 4,893 & 0.09 \\
Yorkshire and The Humber & 68,703 & 214,932 & 7,913,376 & 4,669 & 0.74 \\
East of England & 53,374 & 202,474 & 5,718,764 & 3,614 & -1.12 \\
East Midlands & 37,521 & 145,982 & 4,353,188 & 3,078 & -0.84 \\
West Midlands & 39,390 & 168,555 & 4,824,586 & 3,029 & -0.84 \\
Wales & 32,660 & 137,766 & 4,156,235 & 2,647 & -0.59 \\
North East & 25,053 & 115,418 & 2,927,627 & 1,787 & -0.49 \\
\midrule \bfseries Total & \bfseries 830,770 & \bfseries 1,239,471 & \bfseries 109,160,492 & \bfseries 38,983 & \bfseries 0.00 \\
\bottomrule
\end{tabular}
\end{table}



Here we can see that the least cohesive region is the South West, while the most cohesive is London. The North West also appears to have less cohesive LADs, with a similar value to the South West, while Yorkshire is the second most cohesive below London.

::: {.cell execution_count=7}

::: {.cell-output .cell-output-display}
![Average transformer vector associated with each location corpus coloured by K Means clusters where $K=5$. (A) PCA decomposed into 2 dimensions. (B) Geographic location of clusters.](main_files/figure-pdf/fig-clusters-output-1.pdf){#fig-clusters}
:::
:::


@fig-clusters (A) shows clusters for transformer embeddings decomposed into two dimensions, indicating LAD embeddings that share similar semantic properties. @fig-clusters (B) and (C) show that clusters appear to broadly indicates three distinct regions within the UK, where cluster 0 most closely identifies with Scotland and Wales, 1 with England, and 2 with London and surrounding areas. The few areas that appear as cluster 1 in Wales and Scotland appear to be major urban centres like Cardiff, Glasgow, and Edinburgh. Generally, clusters appear clearly geographically restricted, and even broadly conform with administrative regions like the Welsh and Scottish borders.

## Moran's I

::: {.cell execution_count=8}

::: {.cell-output .cell-output-display}
![Moran's I Plot: LAD embeddings decomposed into 1 dimension and standardised against their spatial lag.](main_files/figure-pdf/fig-morans-output-1.pdf){#fig-morans}
:::
:::


We explore the global spatial autocorrelation of our embeddings by plotting our single decomposed values against their spatial lag on @fig-morans. A Moran's I value of 0.34 indicates a reasonably strong spatial autocorrelation, confirming that semantic information often similar with nearby locations. Given spatial autocorrelation is a univariate analysis, we decompose our high dimensional embeddings down to a single dimension using principal component analysis, leading to an explained variance of 22%. While this does suggest that a large proportion of semantic information is lost in this decomposition, we believe it still provides usable information, particularly given it is very unlikely to demonstrate geographic autocorrelation at random chance.

::: {.cell execution_count=9}

::: {.cell-output .cell-output-display}
![Local Indicators of Spatial Auto-correlation (LISA). (A) Raw plottted 1-dim embeddings. (B) Local Moran's I values ($Is$). (C) LISA HH and LL significant values ($p<0.05$), both are included as the value of embeddings do not convey information.](main_files/figure-pdf/fig-lisa-output-1.pdf){#fig-lisa}
:::
:::


To explore local indicators of spatial autocorrelation (LISA) we plot each decomposed embedding on @fig-lisa (A), each local Moran's I value on (B) and all significant ($p<0.05$) HH and LL LISA quadrants on (C). Note that only selecting significant p values ensures that no regions are included that have values that could demonstrate autocorrelation even if randomly distributed geographically. From this figure, we can see that areas with notably strong spatial correlation include;

* northern Scotland
* west Wales
* the South West; towards Cornwall
* Liverpool, Manchester and northern areas in the North West
* London and surrounding LADs

Non-significant values however do include regions that could be expected to similarly demonstrate spatial autocorrelation, primarily the North East of England, urban areas in Scotland and larger parts of Wales.

NOTE: Interestingly, when we compare the first and second UMAP dimension they capture different semantic information. The first dimension captures a distinction between London and other locations, while the other captures a distinction between London, Scotland, and Wales against the rest of the UK. Values show semantic clusters, regions with significantly similar neighbours etc.

::: {.cell execution_count=10}

::: {.cell-output .cell-output-display}
![Cosine similarity of embeddings for administrative regions across the UK. Higher values indicate greater cosine similarity. Regions shown in descending order by mean cosine similarity value.](main_files/figure-pdf/fig-similarity-output-1.pdf){#fig-similarity}
:::
:::


<!-- ```{python} -->
<!-- # | label: fig-similarity -->
<!-- # | fig-cap: "NOTE:" -->
<!---->
<!-- from paper.figures import plt_cosine_heatmap -->
<!---->
<!-- plt_cosine_heatmap(region_embeddings) -->
<!-- plt.show() -->
<!-- ``` -->

## Semantic Similarity of Regions

On @fig-similarity we compare the cosine similarity between each regional embedding, allowing for inter-regional cohesion to be explored. Both the North and South West have the overall highest levels of cosine similarity with each other region, displaying comparatively high similarity with each other region. London has the lowest overall similarity, only sharing high cosine values with the South and South East of England. This observation is perhaps indicative of an alternative North-South divide that is often considered in England, which from a semantic context appears to be more related to proximity to London. However, a broader but more subtle pattern does appear to reveal distinction between London, the South, and South East. These regions have lower similarity to the Midlands and North of England, which conforms with a typical view of the English North-South divide [@jewell1994].

As expected, Scotland and Wales have low overall cosine values, with Wales lower with respect to London. The total values show a clear pattern where the least cohesive regions appear to be London, Wales, and Scotland.

## Regional Identities

@fig-identity shows the distribution of regional identities identified through our zero-shot classification methodology. Across each region, the 'English' identity is always lower than 'British', suggesting that regions within England typically identify more strongly with the United Kingdom than solely England. Unlike English regions however, both Scotland and Wales identify most strongly with their respective nationalities. While the Welsh identity is reasonably weak, below the weakest British identity in the UK, the Scottish identity is by far the most strong of any identity across every region. Additionally, London has a distinctly higher average confidence value of both British and English identities compared with all other regions. These three regions have the lowest overall cosine similarity values indicated on @fig-similarity (Total), perhaps influenced by these distinct differences in identity.

These results broadly mirror those of past work considering regional identities through the use of structured surveys. Scottish and Welsh identities are both considered stronger than a British identity within these countries, but the Scottish identity is much stronger than the Welsh identity [@haesly2005].

::: {.cell execution_count=11}

::: {.cell-output .cell-output-display}
![Zero Shot classification of each corpus into regional identities; [B]ritish, [E]nglish, [S]cottish, [W]elsh. Values show mean confidence value across each comment, lines indicate 95% confidence intervals. Descending order by [B]ritish confidence.](main_files/figure-pdf/fig-identity-output-1.pdf){#fig-identity}
:::
:::


To identify why the model makes certain predictions, we input a selection of phrases to identify the model confidence outputs, shown on Table \ref{tbl-interpret}. The model makes some interesting choices depending on the sentiment of the input text. For example if we input 'I love Scotland!' the model labels the text as Scottish, while if we change the text to 'I hate Scotland!' the model labels the text as British, with similar results of Wales. These results perhaps suggest that the model understands that personal national sentiment is typically more likely to be positive. 'I hate England!' and 'I love England!' are both considered British or English, but with low confidence. In many cases however, the model appears mostly reluctant to label any text explicitly talking about English locations or topics as English, and prefers to use British. This perhaps reflects the general strong association of England with the UK identity, which is not found with either Scotland and Wales.

\begin{table}[tb]
\centering
\caption{Model response to strong national preferences.}
\label{tbl-interpret}
\begin{tabular}{llr}
\toprule
Sentence & Label & Confidence \\
\midrule
I hate Scotland. & British & 0.54 \\
I love Scotland. & Scottish & 0.97 \\
I hate Wales. & British & 0.58 \\
I love Wales. & Welsh & 0.99 \\
I hate England. & British & 0.48 \\
I love England. & English & 0.53 \\
I live in Manchester. & British & 0.48 \\
I live in Glasgow. & Scottish & 0.64 \\
\bottomrule
\end{tabular}
\end{table}



# Conclusion {#sec-conclusion}

Our paper demonstrates the ability to compare aggregate semantic information for local authorities and regions within the UK, from Reddit comments that mention geoparsed locations. We demonstrate the ability to quantitatively compare semantic similarities between these regions, identifying cohesive meta-regions with similar semantic properties, and regions which appear semantically distinct from the rest of the UK, including London, Wales, and Scotland. To demonstrate the national identity associated with regions, we implement a zero-shot classification, which reflects strong national identity in Scotland, with a weaker national identity in Wales. These national identities are likely contributing factors in the relative semantic isolations demonstrated when clustering LAD embeddings or examining the cosine similarity of regional embeddings in these countries.


Unlike previous work, we do not rely on geotagged social media data, demonstrating regional identity from general, informal text directly associated with locations, rather than from the perspective of dialects. Our results appear to broadly conform with general studies using both mobility data, survey, and geotagged social media, where distance and regional identities appear to influence the distribution of embedding similarities.

Given geoparsing methods enable a geographic dimension to non-geotagged social media data, a much larger repository of informal natural language geographic text is available for research. Future work may consider the use of Reddit comment data to derive notable urban areas of interest for example [@chen2019]. This area of research in particular would benefit from methodologies focussing on the extraction of fine-grained locations from text, which at present is a challenging task [@han2018].

# References {-}

