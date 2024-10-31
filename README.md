# Large Language Models Assignment 2


## Text Classification Models

The Goal is to build a text classification model that will predict whether a piece of text is *positive*, *negative* or *neutral*

This Experiment uses the [**Unified Multilingual Sentiment Analysus Benchmark dataset**](https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual) the models used in this experiment are **only trained on English examples** from the dataset.

**Dataset**

* training set: 1840 instances
* validation set: 325 instances (used as test set for model performance)

**The language these models are trained on is often explicit and unfiltered internet content**

## Models Used in this Experiment

[Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
[Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)
[Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)