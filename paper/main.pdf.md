---
title: "Mapping Semantic Regional Variation in Great Britain through Reddit Comments"
engine: jupyter
execute:
    enabled: true
    echo: false
    warning: false
    cache: false
author:
  - name: Anonymous
format:
  arxiv-pdf:
    keep-md: true
    linenumbers: true
    doublespacing: true
  docx:
    number-sections: true
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

Observed regional variation in geotagged social media text is often attributed to dialects, where features in language are assumed to exhibit region-specific properties. While dialects are seen as a key component in defining the identity of regions, there are a multitude of other geographic properties that may be captured within natural language text. In our work, we consider locational mentions that are directly embedded within comments on the social media website Reddit, providing a range of associated semantic information, and enabling deeper representations between locations to be captured. Using a large corpus of Reddit comments from UK related local discussion subreddits, we first identify place names using a transformer-based named entity recognition model. Embedded semantic information is then generated from these comments and aggregated into local authority districts, representing the semantic footprint of these regions. These footprints broadly exhibit spatial autocorrelation, with clusters that conform with the national borders of Wales and Scotland. London, Wales, and Scotland also demonstrate notably different semantic footprints compared with the rest of the UK, partially explainable through the perception of national identity associated with these regions.



# Introduction

While formal geographic regions within Great Britain are typically designed for administrative and political purposes, they are non-restrictive in how populations can move between them. The level of geographic cohesion between regions across the UK is often studied from the context of mobility, where data sources like Census or transport records describe the physical movement of populations and individuals across geographic space [@rae2009;@titheridge2009], or through non-physical networks using phone records [@sobolevsky2013;@reades2009;@zheng2015;@lambiotte2008], and social media [@lengyel2015;@arthur2019;@sui2011]. When these networks are examined, cohesive clusters develop, which broadly appear to correlate with administrative boundaries [@arthur2019;@ratti2010].

As with social interactions, language is also known to exhibit regional variation [@trudgill2004], and certain dialects contribute to the concept of identity in regions across the UK [@haesly2005;@llamas2014;@llamas2009]. While variations in dialects may be explored from the perspective of survey data [@clopper2016], alternative approaches have considered statistical methods for determining lexical variation in geographically bounded social media networks [@russ2012;@doyle2014;@huang2016;@goncalves2014;@perez2019;@arthur2019;@eisenstein2014]. Comparative analysis of corpora associated with geotagged locations have similarly been shown to exhibit regional properties; for example, Tweets originating in the North East of England are noticeably different compared with the South [@arthur2019].

While social media text has been shown to exhibit regional variation, these works prevalently rely on the use of geotagged posts. The content of geotagged posts is often unrelated to the geotagged location [@kropczynski2018], meaning any geographically associated lexical variation is only attributable to dialects, despite only presenting a single component of language that may exhibit geographic heterogeneity. Instead, our paper explores the similarity of corpora with respect to locational mentions taken from social media text, embedded directly within a related semantic context. Place names may be attributed with coordinate information through a process called geoparsing, where names are identified automatically in text through a natural language processing technique called named entity recognition, and associated an entry in a gazetteer. This approach offers an alternative perspective for the analysis of geographic variation in social media, built directly from the semantic information associated with locations, rather than the location associated with social media users themselves. Semantic information therefore represents a broad range of topics and associations directly relating to geographic locations, rather than just the dialect of contributing users.

Our hypothesis is that related semantic information regarding locations from social media embeds a regional semantic footprint, representing the collective geographic knowledge of each individual user in our corpus. Additionally, any geographic knowledge extracted from social media represents an informal, place-based understanding of geography, which more appropriately reflects a cognitive understanding and perception of locations [@sui2011;@goodchild2011;@berragan]. Semantic footprints are represented through embeddings; numerical representations of text that capture semantic information, enabling a deeper analysis compared with lexicons, and allowing for a direct numerical comparison between them. To explore why certain regions in the UK appear more semantically isolated, we consider the link between identity, querying a large language model to determine the perception of the strength of national identities in UK regions.

@sec-literature first gives an overview of work exploring semantic variation in social media text, place-based geographic knowledge, and regional identities. @sec-methodology first describes our data, then outlines the processing used to generate regional embeddings for comparison and our method for identifying regional identities within these comments. @sec-results presents our results and @sec-conclusion concludes with suggestions for future work.

We have identified the following key aims that are addressed in this paper:

> **Aim 1:** Generate cohesive clusters of regions in Great Britain based on embedded locational semantic information.

> **Aim 2:** Calculate Semantic similarity measures between regional corpora to determine the least and most semantically connected regions.

> **Aim 3:** Examine the distribution of perceived national identity across all British regions using a zero-shot classification.

# Geographic Variation in Social Media Text {#sec-literature}

While it is well understood that despite the advancement of communication technologies, social ties still appear to be geographically restricted [@onnela2011;@sobolevsky2013;@ratti2010;@scellato2010;@takhteyev2012], it is less well understood why these geographic restrictions appear to correlate with administrative boundaries [@ratti2010;@arthur2019;@stephens2015;@yin2017a]. When considering the lexical properties of text associated with geotagged social media data, geographically cohesive groupings occur [@huang2016;@eisenstein2014;@goncalves2014;@arthur2019], which conforms with the idea that dialects are an important component in the identity of regions [@haesly2005;@llamas2014;@llamas2009]. Dialects however, only present a single component of language that may be associated with geographic locations.

The prevalence of social media data for use in geographic research has generated a renewed interest in the concept of 'place' [@wagner2020;@purves2019;@westerholt2018a;@berragan], as contributions to social media are theorised to capture informal knowledge that represents a 'place-based' understanding of geography [@goodchild2011;@sui2011]. In the context of language, this place-based knowledge is generated through 'vernacular geography', which describes the natural language used when informally describing geographic locations [@gao2017a;@goodchild2011;@waters2003;@hollenstein2008]. This informal knowledge incorporates associated biases regarding locations, better representing human associations between locations, compared with formal administrative aggregations. In this sense, associations of geography drawn from social media capture place through a 'bottom-up' approach, building knowledge through experience rather than administrative formalisations [@agnew2005;@sui2011]. While many works consider the formalisation of place through social media data, few have considered how the semantic properties of text may reveal associations between regions, captured through vernacular geography.

## Measuring Regional Variation in Language

Regional variations in linguistic characteristics are often quantified from a statistical analysis of lexicons, considering that regions exhibit subtle differences in dialects that may be captured. For example, studies have considered regional differences in common words on Twitter [@huang2016;@eisenstein2014;@goncalves2014;@arthur2019;@russ2012;@han2012;@doyle2014;@zheng2018], or identified spoken differences in dialects from active participation methods [@clopper2016]. While few explicitly consider a comparative analysis of regional dialects, cosine similarity measures have been used to examine geotagged lexicons from Twitter within England and Wales [@arthur2019]. Results demonstrated that the Welsh lexicon appeared to be the most dissimilar to all other regions, while Northern English regions were more similar to each other compared with Southern regions. Alternatively vague 'vernacular' regions have been defined through an analysis of dialects, where geotagged lexical alternations in Tweets were used to establish a clear North and South divide in the USA [@huang2016].

Given dialects are considered key in defining regional identities, these observations likely in part explain the coherent regions that form through both physical and online social interactions. Users are more likely to interact with others that share a similar identity, which is in part captured through the lexical components of language. Despite this, there are a multitude of other properties that contribute to the perceived identity of geographic regions, which are broadly definable as non-specific cultural associations [@middleton2008;@haesly2005]. Studies that consider dialects on social media are limited by the ability to capture geographic properties from existing data sources, where the only geographic component is an associated geotag. Given the content of geotagged social posts does not necessarily relate to the geotagged location, semantic information extracted therefore represents a broad range of non-specific and unrelated topics. Any observed regional variation is only attributable to the dialect of the contributing author, with the assumption that the author is a resident in the geotagged location.

With the ability to accurately geoparse text, locations can instead be extracted that are embedded directly within a related semantic context, without relying on geotags. This method enables semantic information in text to be directly associated with defined geographic locations, providing a context covering a broad range of topics relating to locations, built from the experiential geographic knowledge of contributing authors. Given social media contributes informal geographic knowledge [@sui2011], observations generated from this data conforms with a general understanding of 'place', incorporating cultural knowledge and characteristics, alongside cognitive biases and perceptions of locations. 

In this work we aim to generate a new comparative measure between regions in the UK through an examination of text associated with locations, extracted from comments on the social media website Reddit. While past work has examined variation between regions from the perspective of social media networks, or by examining lexicons associated with geotagged social media messages, we examine regional variations derived from text associated with directly embedded place names. Unlike using geotags, which ascribe linguistic features such as dialect to specific locations, our method instead captures any comment that mentions a location alongside its context. Quantified information therefore does not reflect dialects associated with locations, but common semantic associations, embedding cultural information, or location specific topics and opinions.

## Embedding Semantic Footprints

Statistical comparisons between two or more distinct texts first relies on an appropriate method for processing the text into a numerical format. Typically, a TF-IDF approach is used to generate document embeddings [@daniel2007], which assigns word importance based on the frequency of mentions within a corpus. TF-IDF however does not have the capability to capture broader semantic information, given there is no knowledge of the meaning behind words. Large Language Models (LLMs) instead are pre-trained on a very large corpus of natural language text, which, alongside their architecture, enables them to more appropriately consider semantic information [@devlin2019]. Similarly to TF-IDF, text is input into these models and output as a numerical representation, which embeds words as high dimensional vectors, capturing contextual semantic information. 

This approach differs from past work that only considered a lexical analysis, where semantic information and context is not preserved, instead building vectors that act as semantic representations of locations identified in our corpus, which we name 'semantic footprints'. Given semantic information is preserved, locational embeddings are able to reflect the deeper associations between geographic locations, built from a multitude of contexts and perspectives, forming an aggregate representation. Any geographically cohesive relationships between footprints therefore demonstrate a direct association between geography and language, which hasn't been captured previously.

## Zero Shot Classification

As LLMs are pre-trained on a large corpus of natural language text, they build representations of this text that emulate a human understanding of language. In theory these representations capture the collective knowledge of humans that contributed the natural language text used to build them. Therefore, in addition to factual information, when posed with non-deterministic questioning, these models are able to contribute the biased information that is incorporated into their model weights.

Recent research has noted on the ability to perform zero-shot classification using LLMs, where class predictions may be made without the model ever having previously seen the labels [@wei2022a;@wei2022]. While research has considered the use of questionnaires to query the strength of national identities within the UK [@haesly2005;@griffiths2022], an LLM may instead be used. For example, an LLM may be questioned whether it personally feels a sequence of text appears to be 'British', 'English', 'Scottish', or 'Welsh'. Through this zero-shot classification, we are able to determine the perceived strength of national identity associated with each region in our work, to examine whether this appears to correlate with any cohesion between the semantic footprints that we generate.   

# Methodology {#sec-methodology}

The following section first introduces our main data source; the social media website Reddit, from which we access a collection of user submitted comments. Following this, we detail our methodology for generating embeddings from each of these comments, and the processing that follows, to analyse the geographic properties of these embeddings.



[Reddit](https://reddit.com) is a public discussion, news aggregation social network, and among the top 20 most visited websites in the United Kingdom. In 2020, Reddit had around 430 million active monthly users, comparable to the number of Twitter users [@murphy2019;@statista2022]. Reddit is divided into separate independent _subreddits_ each with specific topics of discussion, where _users_ may submit _posts_ which each have dedicated nested conversation threads that users can add _comments_ to. Subreddits cover a wide range of topics, and in the interest of geography, they also act as forums for the discussion of local places. The [United Kingdom subreddit](https://reddit.com/r/unitedkingdom) acts as a general hub for related topics, notably including a list of smaller and more specific related subreddits. This list provides a 'Places' section, a collection of local British subreddits, ranging in scale from country (`/r/England`), region (`/r/thenorth`, `/r/Teeside`), to cities (`/r/Manchester`) and small towns (`/r/Alnwick`). In total there are 213 subreddits that relate to 'places' within the United Kingdom^[https://www.reddit.com/r/unitedkingdom/wiki/british_subreddits]. We use the corpus generated by @berragan, which consists of a collection of all Reddit comments taken from each UK related subreddit [@baumgartner2020], with place names identified by a custom transformer-based named entity recognition model^[https://huggingface.co/cjber/reddit-ner-place_names]. In total 8\,282\,331 comments were extracted, submitted by 490\,535 unique users, between 2011\-01\-01 and 2022\-04\-17. Table \ref{tbl-example} gives an example entry from this geoparsed Reddit corpus.

There are a total of 52\,169 unique locations in this corpus, with a highly skewed distribution in mentions. Most locations were only mentioned a single time, while 'London' was mentioned in almost 300,000 comments. To reduce this skew, we sampled any location mentioned more than 5,000 times, retaining only up to 5,000 randomly sampled comments per location. The goal with this processing was to ensure that our generated embeddings did not simply become biased towards the word embedding for a single location, and instead capture a broader sense of an aggregate region. In our data subset, we find that 1% of users (1\,698) mention 29% of our place names.

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





Table \ref{{tbl-sum}} gives an overview of the number of comments, word count and number of places that were identified within each administrative region of the UK. Our study concerns a subset of the full Reddit comment dataset, subsetting each location to a maximum of 5,000 mentions, leaving a total of 830\,770 comments containing place names. Comments range from 1 to 3\,555 words in length, with a mean length of 79. 

\begin{table}[tb]
\centering
\caption{Summary of comments relating to each region in our study.}
\label{tbl-sum}
\begin{tabular}{lrrrr}
\toprule
RGN21NM & Total Comments & Unique Words & Word Count & Total Places \\
\midrule
Scotland & 181,831 & 437,746 & 23,218,279 & 8,052 \\
South East & 107,134 & 308,491 & 11,849,441 & 5,679 \\
London & 206,280 & 422,036 & 23,868,430 & 5,164 \\
South West & 85,960 & 267,388 & 9,680,548 & 5,090 \\
North West & 88,789 & 259,154 & 10,650,018 & 4,893 \\
Yorkshire and The Humber & 68,703 & 214,932 & 7,913,376 & 4,669 \\
East of England & 53,374 & 202,474 & 5,718,764 & 3,614 \\
East Midlands & 37,521 & 145,982 & 4,353,188 & 3,078 \\
West Midlands & 39,390 & 168,555 & 4,824,586 & 3,029 \\
Wales & 32,660 & 137,766 & 4,156,235 & 2,647 \\
North East & 25,053 & 115,418 & 2,927,627 & 1,787 \\
Total & 830,770 & 1,239,471 & 109,160,492 & 38,983 \\
\bottomrule
\end{tabular}
\end{table}



## Generating Embeddings

We first generate semantic embeddings for each comment in which a location was mentioned, using the `sentence-transformers` Python library [@reimers2019], with the `all-mpnet-base-v2` model. With our selected embedding model, we then performed the following steps to generate embeddings for each Local Authority District (LAD) in Great Britain.

1. Masked any place name with a generic token: 'PLACE'.
2. Generate sentence embeddings for each comment.
3. Group embeddings by LAD using identified locations, taking the mean embedding.

To visualise the outputs from this processing we consider an example comment $s_1 = \text{"I live in London."}$, shown on Equation \ref{eq-dims}. 

$$
\begin{aligned}
\mathit{s_{i}} &= \text{'I live in \textit{London}'} \\
\textbf{1. }\downarrow \\
\mathit{s_{i}} &= \text{'I live in \texttt{PLACE}'},
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
\textbf{3. }\mathit{LAD_{j}} = 
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

In Equation \ref{eq-dims}, $n$ is the `sentence-transformers` embedding dimension (768), and $t$ is the total number of unique comments that relate to a single LAD region ($LAD_j$). Values ($x_i$) in step **2.** are model weights that represent the embedding for the comment $s_i$, capturing semantic information. All comment embeddings associated with $LAD_j$ are then processed into one dimension by taking the mean (step **3.**), producing the semantic footprint. Reducing embedding dimensionality by taking the mean is common in NLP tasks. For example, sentence embeddings are simply the average of word embeddings, and are used in many tasks like topic identification or sentence similarity, with good results [@reimers2019].

By masking place names we ensure that no comment embeddings accidentally incorporated geographically grounded information. For example, comments in South Eastern local authorities are likely to frequently mention London, given they are geographically close. Embeddings for these locations would therefore capture an association through the mention of London, rather than general semantic information. For our work, we want to exclude any geographic information, ensuring that embeddings solely capture semantic associations.

Given that transformers are a fairly new architecture in natural language processing, and the creation of these models require significant computational resources and training time, their use to date has been limited in related research. Our choice to use the transformer architecture stems from the emphasis we place on the extraction of nuanced and contextual semantic information, which is lost with more statistical methods. It should be noted however that while TF-IDF lexical methods are less complex, they are typically more interpretable; for instance, words that contribute importance to an embedding may be extracted from a TF-IDF model. The numerical representations of any text generated by transformers are not directly interpretable in this manner.

## Spatial Clustering and Autocorrelation

It is reasonable to assume that there are LADs within our corpora that generate embeddings that share similar properties. A typical method to group unlabelled multi-variate data based on shared properties uses unsupervised clustering. Therefore, to explore whether geographically cohesive clusters appear within our semantic embeddings (Aim 1), we generate hierarchical clusters, which are non-geographically bounded.

Agglomerative clustering was used to generate clusters of LAD embeddings, a type of hierarchical clustering method which we found performed better compared with K Means. Agglomerative clustering is able to automatically select the optimal number of clusters based on a distance threshold, which in our data was determined to be 1.5, giving three clusters. These Clusters were visualised geographically to examine whether geographically cohesive groupings occurred. The proportion of clusters presented within each administrative region (RGN)^[The highest tier of sub-national division in England. For Scotland and Wales we use the full national extents.] in Great Britain was also plotted to determine whether clusters appeared to correlate with administrative boundaries.

To further explore aim 1, we consider whether embeddings demonstrate spatial autocorrelation through the Moran's I metric, and identify geographic clusters of shared semantic attributes, using a Local Indicators of Spatial Autocorrelation (LISA) analysis [@anselin1995;@rey2023]. As spatial autocorrelation analysis is only univariate, we consider embeddings UMAP decomposed into 2 dimensions for this analysis, rather than the original 768 dimensional embeddings. It is important to note that the numerical value of this decomposed embedding does not convey any definable information based on its magnitude, meaning unlike typical LISA analysis, HH and LL values are only distinguished to highlight non-specific differences in semantic information.

### Semantic Similarity

We determine the semantic similarity between regions across Great Britain (Aim 2), by grouping LAD embeddings into their respective RGN, taking the mean embedding value across each region. We then calculate the cosine similarity between each and every embedding, demonstrating the inter-region cohesion across Great Britain.

Cosine similarity is a common metric for comparing embeddings, as it is invariant to the magnitude of the vectors, and only considers the direction. This is important as the magnitude of embeddings is not meaningful, and only the direction of the vector conveys information. For example, the embedding for 'London' cannot be twice as important as the embedding for 'Manchester', and therefore the magnitude of the embedding is not meaningful.

## Zero-Shot national identities

To determine whether the distribution of identities correlate with our findings (Aim 3), we examine the relative national identity strength based on the semantic information embedded within comments. Semantic information is expected to capture both explicit information contributed by users; for example stating 'London is a British city', in addition to implicit semantic information that exists within language. For this reason, we do not mask place name mentions in these embeddings. For example the phrase 'bonnie Scotland' may suggest a strong identity due to the inclusion of Scottish slang^[See 'Scottish English' or 'Scots'; [@stuart-smith2008]].

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

# Results {#sec-results}

::: {.cell execution_count=6}

::: {.cell-output .cell-output-display}
![Average transformer vector associated with each location corpus coloured by hierarchical Agglomerative clusters where $K=3$. (a) LAD embeddings UMAP decomposed into 2 dimensions. (b) Proportion of clusters by RGN. (c) Geographic location of clusters.](main_files/figure-pdf/fig-clusters-output-1.pdf){#fig-clusters}
:::
:::


@fig-clusters (a) shows clusters of LAD transformer embeddings UMAP decomposed into two dimensions, indicating embeddings that share similar semantic properties. These clusters appear to broadly correlate with three distinct regions within the UK, where cluster 0 most closely identifies with Scotland and Wales, 1 with England, and 2 with London and surrounding areas (@fig-clusters (b-c)). The few areas that appear as cluster 1 in Wales and Scotland are major urban centres like Cardiff, Glasgow, and Edinburgh, in addition to Pembrokeshire in Wales. Overall these clusters appear to be geographically restricted, and even broadly conform with administrative regions like the Welsh and Scottish borders.

These findings appear to correlate with past work that has observed strong 'boundary effects', where lexical similarity between geotagged Tweets often correlates with administrative boundaries [@li2021;@bailey2018;@arthur2019;@yin2017a]. This observation in our results appears particularly strong at the borders of Scotland and Wales, suggesting that the prominent sense of identity within these countries has generated semantic coherence [@haesly2005]. Note that unlike past work that has demonstrated this effect from the perspective of dialects or interviews from residents, our results instead capture the perceived sense of identity associated with these countries, built from all users in our corpus.

As Glasgow, Edinburgh and Cardiff share a cluster with English LADs rather than their respective country, these locations are more semantically connected with the rest of the UK, compared with other regions. This observation mirrors the results of work that considered co-occurring locational mentions between cities, where shared city mentions in text often appear irrespective of distance, and across administrative borders [@berragan]. However, while Pembrokeshire is not a major urban centre, the alignment with English LADs is likely due to its interest as a holiday destination. 

Cluster 2 generated surrounding London suggests a distinct perceived separation with this region of the UK. This is interesting given London's extensive physical connections through high speed rail, and general sense of strong association with other cities, given it is the capital city [@berragan]. Our results therefore suggest that despite London's importance nationally, semantic information is able to capture a deeper context that dissociates it from other regions in the UK.

The following section formalises the level of geographic coherence our embeddings exhibit, and explores the key locations that drive the relationship between text and geography.

## Moran's I Analysis

::: {.cell execution_count=7}

::: {.cell-output .cell-output-display}
![Moran's I Plot: LAD embeddings decomposed into 2 dimensions and standardised against their spatial lag.](main_files/figure-pdf/fig-morans-output-1.pdf){#fig-morans}
:::
:::


To quantify whether our embeddings demonstrate spatial autocorrelation, we consider the Moran's I metric, which considers the spatial relationship between each observation and its neighbours [@anselin1995]. Given this analysis requires univariate data, we explore global spatial autocorrelation of our UMAP decomposed embeddings by plotting both dimensions against their spatial lag on @fig-morans. The Moran's I values of <esda.moran.Moran at 0x7ef697c90310> and <esda.moran.Moran at 0x7ef697c90280> indicate a reasonably strong spatial autocorrelation with both embedding dimensions, confirming that semantic information is often similar with nearby locations. While the Moran's I values for both dimensions are similar, their cosine similarity is low (array([[-0.09880356]], dtype=float32)), meaning it is likely that these two decomposed dimensions capture distinctly different semantic traits.

While spatially coherent results have been demonstrated from the perspective of dialects on social media [@russ2012;@doyle2014;@huang2016;@goncalves2014;@perez2019;@arthur2019;@eisenstein2014], we have demonstrated that this phenomenon can also be captured from general semantic information. Dialects have always been considered to have strong geographical grounding [@trudgill2004], but unlike dialects, it is more surprising that general semantic information regarding locations similarly exhibits this relationship.

::: {.cell execution_count=8}

::: {.cell-output .cell-output-display}
![Local Indicators of Spatial Auto-correlation (LISA). (a/d) 1 dimensional embedding values. (b/e) Local Moran's I values ($Is$). (c/f) LISA HH and LL significant values ($p<0.05$), both are included as the value of embeddings do not convey information.](main_files/figure-pdf/fig-lisa-output-1.pdf){#fig-lisa}
:::
:::


To explore local indicators of spatial autocorrelation (LISA) we plot each decomposed embedding on @fig-lisa (a/d), each local Moran's I value on (b/e) and all significant ($p<0.05$) HH and LL LISA quadrants on (c/f). Note that only selecting significant $p$ values on @fig-lisa (c/f) ensures that no regions are included that have values that could demonstrate autocorrelation even if randomly distributed geographically. From @fig-lisa (c/f), we can see that notable areas with strong spatial correlation include;

* Scotland
* Wales
* London and surrounding LADs
* the South West; towards Cornwall

When we compare the first and second UMAP dimension they appear capture different semantic information. London for example only appears in dimension 1, while dimension 2 captures more broad spatial autocorrelation across Scotland and Wales. In Scotland we can see that from the dimension 1 LISA, both Glasgow and Edinburgh represent areas of HL/LH, where semantic information in these cities is not the same as surrounding LADs, and effect that is also captured in some LADs surrounding London. For England, there appears to be fewer geographically cohesive semantic associations, most notably in the South West and part of the Midlands.

These results again demonstrate the geographically cohesive and administrative boundary effects that have been observed in past work, with respect to Scotland and Wales. While in addition we have captured a notable grouping in the South West, which potentially reflects the Cornish identity [@deacon2007], and a grouping associated with London.

## Semantic Similarity of Regions

::: {.cell execution_count=9}

::: {.cell-output .cell-output-display}
![Scaled cosine similarity of embeddings for administrative regions across the UK. Higher values indicate greater cosine similarity. Regions shown in descending order by mean cosine similarity value.](main_files/figure-pdf/fig-similarity-output-1.pdf){#fig-similarity}
:::
:::


On @fig-similarity we compare the cosine similarity between each RGN embedding, allowing for inter-regional cohesion to be explored. Both the North and South West have the overall highest levels of cosine similarity with each other region, displaying comparatively high similarity with each other region. London has the lowest overall similarity, only sharing high cosine values with the South and South East of England. As expected, Scotland and Wales have low overall cosine values, with Wales sharing even lower similarity with respect to London compared with Scotland. Mean values show clearly that the least cohesive regions appear to be London, Wales, and Scotland, three regions that are also those with the strongest levels of spatial autocorrelation. 

Excluding London, the North East is the region in England with the lowest overall cosine similarity with the rest of the UK. This is perhaps reflective of the sense of inter-region identity that is often noted with this region [@middleton2008], as well as the lack of physical interaction with the rest of the country due to a reduction in local industry and jobs [@kalantaridis2010]. Alternatively, given the North West is home to nationally important cities like Manchester and Liverpool, it shares strong similarity across the whole of the UK, excluding Wales. Other patterns emerge in regions like Yorkshire, where there is notably low similarity with the South Western regions of England, and while Wales shares low overall cosine similarities with England, Scotland has a slightly increased similarity, particularly with London. Major urban centres in Scotland are relatively well connected through rail routes, and Edinburgh and Glasgow are historically important UK cities, while Wales in this sense is less directly associated with the rest of the UK.

## Regional Identities

Finally, we consider whether the concept of a perceived regional identity generated from a large language model aligns with the observed regional variation in our analysis. @fig-identity shows the distribution of regional identities identified through our zero-shot classification methodology.

Across each region, the 'English' identity is always lower than 'British', suggesting that regions within England typically identify more strongly with the United Kingdom than solely England. Unlike English regions however, both Scotland and Wales identify most strongly with their respective nationalities. While the Welsh identity is reasonably weak, below the weakest British identity in the UK, the Scottish identity is by far the most strong of any identity across every region. Of the English regions, London has a distinctly higher average confidence value of both British and English identities compared with all other regions. These results also broadly agree qualitative interviewing, where the national identity of both Scotland and Wales is typically stronger compared with their British identity, but the Welsh identity is typically weaker than the Scottish identity [@carman2014;@llamas2014;@llamas2009;@haesly2005]. Given these three regions have the lowest overall cosine similarity values indicated on @fig-similarity (Mean), these perceived differences in identity are a likely component in their semantic differences.

::: {.cell execution_count=10}

::: {.cell-output .cell-output-display}
![Zero Shot classification of each corpus into regional identities; [B]ritish, [E]nglish, [S]cottish, [W]elsh. Values show mean confidence value across each comment, lines indicate 95% confidence intervals. Descending order by [B]ritish confidence.](main_files/figure-pdf/fig-identity-output-1.pdf){#fig-identity}
:::
:::


To identify why the model makes certain predictions, we input a selection of phrases to identify the model confidence outputs, shown on Table \ref{tbl-interpret}. The model makes some interesting choices depending on the sentiment of the input text. For example if we input 'I love Scotland.' the model labels the text as Scottish, while if we change the text to 'I hate Scotland!' the model labels the text as British, with similar results of Wales. These results perhaps suggest that the model understands that personal national sentiment is typically more likely to be positive. As demonstrated on Twitter for example, negative perceptions of locations are generally directed towards locations external to a users home location [@butler2018]. However, 'I hate England!' and 'I love England!' are interestingly both considered British or English, but with low confidence. In many cases, our model appears mostly reluctant to label any text explicitly talking about English locations or topics as English, and prefers to use British, perhaps reflecting the general strong association of England with the UK identity, which is as strong in either Scotland and Wales. 

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

Our paper demonstrates the ability to compare aggregate semantic information for local authorities and regions within the UK, from Reddit comments that mention geoparsed locations. When examining the semantic properties of each LAD in the UK, we find that geographically cohesive clusters appear, which exhibit a relatively strong spatial autocorrelation. Clusters broadly conform with the national borders of Scotland and Wales, while London also appears to be semantically distinct from the rest of the country. When examining the perception of national identities associated with regions, we find that these identities conform with these distinct geographic groupings.

As demonstrated in past work that has examined both physical and non-physical networks, our observed semantic information similarly appears to correlate with pre-defined administrative boundaries, particularly the national boundaries of Scotland and Wales [@li2021;@bailey2018;@arthur2019;@yin2017a]. Additionally, the distinct difference in footprints between each constituent country in the UK likely relates to the perceived national identity that we generate through our zero-shot transformer classification. Scotland in particular is noted as having a strong national identity that is distinct from Britain, while similar is true in Wales, but to a lesser extent [@haesly2005;@carman2014]. Similar to how dialect and social interaction networks conform with administrative regions, semantic information likely embeds the cultural perception of identity that causes distinct borders between regions. Interestingly however, despite having clear semantic associations, given they share a cluster on @fig-clusters, and generate similar decomposed embedding values on @fig-lisa, both Wales and Scotland are demonstrated to have distinct national identities on @fig-identity.

Despite most locations across Scotland and Wales appearing disconnected with the rest of the UK, major cities like Glasgow and Edinburgh are more semantically similar, a distinction that was also observed when the distance decay of locational co-occurrences in text was examined [@berragan]. This observation suggests that these cities do appear to be typically more semantically cohesive regardless of geographic distance, while other locations typically share semantic properties within the same nation, captured through a stronger spatial autocorrelation.

When considering the cosine similarity between RGN embeddings, we see evidence of an alternative North-South divide that is often considered in England, which from a semantic context appears to be more related to proximity to London. Unlike typical representations of this divide therefore, the South West of England appears to be distinct from the South East, with a stronger association with the North. South Eastern regions however do share lower similarity to the Midlands and North of England, which conforms with a typical view of the English North-South divide [@jewell1994].

Geoparsing methods capture a geographic dimension to non-geotagged social media data, enabling a much larger repository of informal natural language geographic text to be used for research. Future work may consider the use of Reddit comment data to derive notable urban areas of interest for example [@chen2019]. This area of research in particular would benefit from methodologies focussing on the extraction of fine-grained locations from text, which at present is a challenging task [@han2018].

# References {-}

