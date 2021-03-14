# Building a text to emoji sequence translator

## ***This project is in progress***

## Datasets

- [Emoji dictionary](https://www.kaggle.com/eliasdabbas/emoji-data-descriptions-codepoints)

- [Reddit dataset](https://www.kaggle.com/reddit/reddit-comments-may-2015)
    - Number of entries: 54,504,410

## Links

- [Model](https://huggingface.co/sentence-transformers/distilbert-base-nli-stsb-mean-tokens)

## Workflow

1. Parse reddit dataset with 'stsb-distilbert-base' to get sentence encoding for each setence. (Label)

2. Parse reddit dataset with distilbert to get word embeddings for fine-tuning. (Features)

3. Train a model using to fit the labels with input features.

## Model structure

1. A simple transformer layer that gets the features from distilbert and outputs a combination of emoji's (in text form) and gets flattened to form a sentence in word.

2. The formed sentence is passed through 'stsb-distilbert' to fit the label.x
