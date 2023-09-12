<div align="center">

# Mapping Semantic Regional Variation in Great Britain through Reddit Comments

<a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/></a>
<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white"/></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-blueviolet?style=for-the-badge"></a>

</div>

<!--
<p align="center">
<a href="https://cjber.github.io/georelations/src">Documentation</a> •
<a href="todo">FigShare (soon)</a>
</p>
-->

[Cillian Berragan](https://www.liverpool.ac.uk/geographic-data-science/our-people/) \[[`@cjberragan`](http://twitter.com/cjberragan)\]<sup>1\*</sup>,
[Alex Singleton](https://www.liverpool.ac.uk/geographic-data-science/our-people/) \[[`@alexsingleton`](https://twitter.com/alexsingleton)\]<sup>1</sup>,
[Alessia Calafiore](https://www.eca.ed.ac.uk/profile/dr-alessia-calafiore) \[[`@alel_domi`](http://twitter.com/alel_domi)\]<sup>2</sup> &
Jeremy Morley \[[`@jeremy_morley`](http://twitter.com/meremy_morley)\]<sup>3</sup>

<sup>1</sup> _Geographic Data Science Lab, University of Liverpool, Liverpool, United Kingdom_  
<sup>2</sup> _Edinburgh College of Art, University of Edinburgh, United Kingdom_  
<sup>3</sup> _Ordnance Survey, Southampton, United Kingdom_

<sup>\*</sup>_Correspondence_: c.berragan@liverpool.ac.uk

## Abstract

Observed regional variation in geotagged social media text is often attributed to dialects, where features in language are assumed to exhibit region-specific properties. While dialects are seen as a key component in defining the identity of regions, there are a multitude of other geographic properties that may be captured within natural language text. In our work, we consider locational mentions that are directly embedded within comments on the social media website Reddit, providing a range of associated semantic information, and enabling deeper representations between locations to be captured. Using a large corpus of Reddit comments from UK related local discussion subreddits, we identify place names using a transformer-based named entity recognition model. Embedded semantic information is then generated from these comments and aggregated into local authority districts, representing the semantic footprint of these regions. These footprints broadly exhibit spatial autocorrelation, with clusters that conform with the national borders of Wales and Scotland. London, Wales, and Scotland demonstrate notably different semantic footprints compared with the rest of the UK, which may be explainable through the perception of national identity associated with these regions.

## HuggingFace NER Model

The NER model used as part of this work is available on the HuggingFace model hub. Instructions for using this model are included on the model card.

<https://huggingface.co/cjber/reddit-ner-place_names>

## Project layout

```bash
src
├── common
│   └── utils.py  # various utility functions and constants
├── preprocessing.py    # process comments with identified place names
├── embeddings.py    # generate sentence embeddings
└── zero_shot.py  # generate identities using zero shot
```
