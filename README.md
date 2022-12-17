# reddit-ironic-language-detection

This repository is a sklearn implementation of *Sparse, Contextually Informed Models for Irony Detection: Exploiting User Communities, Entities and Sentiment* (Wallace et al., 2015). This paper seeks to fit a binary classification model for verbal irony detection in online content (in this case, Reddit). The fully featurized model with thread context (more on this below) is compared against a baseline bag-of-words (BOW) model.  

## Structure of Dataset

The raw dataset contains the following columns: `comment_id`, `comment	subreddit`, `parent_id`, `label`, `thread_title`. The `subreddit` in this dataset is either Conservative or Progressive. The `label` is 1 for ironic and -1 for not ironic. The `parent_id` is used to traverse up a Reddit comment thread (i.e. retrieve the comment that a user replied to). 

Importantly, only ~5% of the observations in this dataset are labeled as ironic.

## Preprocessing

For the baseline model, comment text is tokenized using the spaCy library. Whitespace and stop words are removed from the text.  

For the fully featurized model, the paper introduces interaction features that capture a combination of proper nouns (NNPs), comment text sentiment, and the subreddit that the comment was found in. For instance, an interaction feature could be Obamacare (the comment thread contains the NNP Obamacare), positive sentiment, and Conservative subreddit. Preprocessing results in a list of strings, where each element in the list are whitespace-separated interaction features that occur in a given comment thread. For instance, 

```
[
  '<sentiment><subreddit><NNP1> <sentiment><subreddit><NNP2> <sentiment><subreddit><NNP3>', 
  '<sentiment><subreddit><NNP1> <sentiment><subreddit><NNP4>',
]
```

To follow these steps, we first use spaCy to retrieve a comment text's polarity (polarity > 0 means positive sentiment, negative sentiment otherwise). The tag '<pos>' indicates positive sentiment and the tag '<neg>' indicates negative sentiment. Similarly, the tag '<cns>' indicates Conservative subreddit and the tag '<lib>' indicates Progressive subreddit. To expand a comment text to the thread text + thread title, we recurse up through the `parent_id` fields and merge the comment text with its parent text (and the parent text with the parent's parent text, and so forth and so on). SpaCy is used to extract NNPs from this text.

Finally, note that the exact floating point polarity scores are kept for each comment text as the paper uses interaction features with polarity scores appended.

## Featurization

A bag of unigrams and bigrams is created for the baseline model. A `DictVectorizer` from sklearn is fit on the training data. Unseen unigrams and bigrams are ignored at test time.

Similarly, a bag of interaction features (with polarity scores appended) is created for the fully featurized model. A binary `CountVectorizer` from sklearn is fit on the training data. Unseen interaction features are ignored at test time. Each feature is then a binary vector with 1s for NNP x sentiment x subreddit combinations that appear in a given row of the preprocessed data (with the polarity score of the comment text appended at the end).

## Model Details and Evaluation 

A logistic regression classifier with SGD training was used to classify examples. The classifier was trained and tested (with a 80-20 train-test split) 50 times to generate 50 precision and recall values. The logistic regression classifier was penalized with L2 regularization and used balanced class weights to adjust weights inversely proportional to class frequencies. A grid search was performed on the alpha parameter to maximize the F1 score.

The final precision values:
| Precision      | Mean  | Median | 25th percentile | 75th percentile |
|----------------|-------|--------|-----------------|-----------------|
| Baseline Model | 0.11  | 0.10   | 0.09            | 0.12            |
| Full Model     | 0.19  | 0.18   | 0.15            | 0.21            |

The final recall values:
| Recall         | Mean  | Median | 25th percentile | 75th percentile |
|----------------|-------|--------|-----------------|-----------------|
| Baseline Model | 0.29  | 0.30   | 0.25            | 0.33            |
| Full Model     | 0.41  | 0.43   | 0.35            | 0.45            |

