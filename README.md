[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/gmihaila/fintech_patents/main/src/fintech_patents/web_app.py)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Fintech Patents Classification

## Intro

This is the work done in the fintech patents classification project. 

Check out our paper: [Identifying FinTech Innovations Using BERT](https://github.com/gmihaila/fintech_patents/raw/main/identifying_fintech_innovations_using_bert.pdf) which was accepted in [BigData2020 IEEE](https://bigdataieee.org/BigData2020/AcceptedPapers.html).

<br>

## Abstract

*Advancements in technology have resulted in the emergence of numerous FinTech innovations. However, a global understanding of such innovations is limited, due to a lack of an underlying taxonomy and benchmark datasets in the FinTech domain. To address this limitation, we develop a FinTech tax- onomy and manually annotate a set of FinTech patent abstracts according to the taxonomy. We use the annotated dataset to train deep learning models, specifically recurrent neural networks and convolutional neural networks combined with state-of-the- art BERT transformers. Experimental results show that the deep learning models can accurately identify FinTech innovations. We use our best performing BERT-based model on a large dataset of financial patent abstracts, and shortlist a set of 25,580 FinTech patent applications submitted to the European and US Patent Offices between 2000 and 2017. We illustrate how an analysis of the shortlisted set can be used to gain understanding of what FinTech innovations are, where and when they emerge, and provide the basis for further work on what their impact is on the companies investing in them, and ultimately on society.*

<br>

## WebApp

There is a webapp that contains most efficient models presented in our paper. 

It is currently live at [share.streamlit.io/gmihaila/fintech_patents](https://share.streamlit.io/gmihaila/fintech_patents/main/src/fintech_patents/web_app.py) 


To run locally: `streamlit run src/fintech_patents/web_app.py`

