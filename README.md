<div align="center">

# Regional Identity and Cohesion Identified in Britain through Reddit Comments

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

Physical and online social interactions broadly decline in strength with geographic distance, and intra-community interaction strength often appears to correlate with pre-defined geographic administrative boundaries. These factors suggest that there are strong regional influences on social interactions, particularly given online social interactions are not directly restricted by distance. Regional identities between constituent countries within the UK are frequently studied from the context of structured surveys and census data, with language seen as a defining feature of these identities. In our work, we consider the exploration of semantic interaction strength, building a comparative analysis of corpora related to regions within the UK, determining shared semantic similarities between them. Building on this analysis, we establish the strength of regional Scottish, Welsh, English, and British identities within each region, using emergent properties of a large language model. We find that distinct groups emerge particularly in the South East and North East of England, while Scottish regional identity appears to be more distinct from England and Britishness compared with the Welsh identity.

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
