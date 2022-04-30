# review-classifier
A naive-bayes classifier trained on the IMDB movie review dataset

# Which File Do I Want?
- ***classifier.py*** - This is the final classifier. It classifies the reviews into either positive (for reviews with scores 7-10) or negative (for reviews with scores 0-5) classes.
- - ***createAndSaveModel*** - Run this to create, train and save the model using Pickle for faster loading when you need it.

# Prerequisites
## IMDB Dataset
Download the IMDB Reviews Dataset [here](https://ai.stanford.edu/~amaas/data/sentiment/)

## .env
classifier.py requires a .env file with the following variables:
- STOPWORDS - Points to stopwords.txt, a list of new-line-delineated stop words (same as the nltk list)
- TRAINPOSDIR - A directory containing all of the positive training reviews
- TRAINNEGDIR - A directory containing all of the negative training reviews
- TESTPOSDIR - A directory containing all of the positive testing reviews
- TESTNEGDIR - A directory containing all of the negative testing reviews
