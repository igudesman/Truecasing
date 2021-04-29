<!-- ABOUT THE Task -->
## About the task

Converting the characters in the company name to the correct case, so that the resulting name most likely coincides with the one prescribed in the charter.

<!-- Research -->
## Research
Truecasing is a NLP problem of finding the proper capitalization of words within a text where such information is unavailable. There are several practical approaches to the Truecasing problem:
* Sentence segmentation: splitting the input text into sentences and up-casing the first word of each sentence. Some heuristics can also be applied (the character after dot also should be capitalized more likely).
* Statistical modeling: by training a statistical model on words and group of words which usually appear in capitalized format. [1]
* Recurrent neural networks: by training a Character-Level BiLSTM-CRF Model. [2]

In this project, only the first two approaches and their combination are considered.


### Statistical modeling

This model requires information on the frequencies of unigrams, bigrams, and trigrams obtained on a training corpus. The idea was taken from "tRuEcasIng" paper with some modifications [1]. Also some code was grabbed from github [3].

### Key aspects of my implementation

The model was trained on the names of Russian companies (data/train.txt). The spaCy library trained on the "ru2_nerus_800ks_96" corpus was used to apply tokenezation, lemmatization and POS tagging.

Training pipeline (train_model.py):
* Tokenize the company name.
* Leave only tokens which are not of this POS: ('NUM', 'PUNCT', 'SYM', 'X', 'SPACE', 'NO_TAG').
* Apply lemmatization for remaining tokens.
* Update unigrams, bigrams, trigrams dictionaries.

Testing pipeline (test_model.py):
* The same steps as for Training pipeline except the last one.
* For each lemmatized token find possible cases and calculate score (the function get_score in the test_model.py). Choose appropriate variant.
* Apply postprocessing for whole company name (Sentence segmentation using heuristics and some and observations from data/train.txt).

The result.txt contains predictions for data/test.txt.
The statistical modele is in the data folder.


### Validation results

The data/train.txt was divided to train and validation sets (5000000 in the train set). The F1-score showed approximately 0.75.


### Further work

Try the Recurrent neural networks approach and compare results with Statistical modeling.

<!-- Requirements -->
## Requirements

* numpy==1.20.2
* pymorphy2==0.8
* spacy==2.3.0
* torch==1.8.1
* tqdm==4.60.0


<!-- Reference List -->
## Reference List
* [1] https://www.cs.cmu.edu/~llita/papers/lita.truecasing-acl2003.pdf
* [2] https://www.aclweb.org/anthology/D16-1225/
* [3] https://github.com/nreimers/truecaser