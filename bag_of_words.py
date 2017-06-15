import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


# nltk.download()


def read_data(file_name):
    train = pd.read_csv(file_name, header=0,
                        delimiter="\t", quoting=3)

    print train.shape
    print train.columns.values

    return train


def process_words(raw_review):
    review_text = BeautifulSoup(raw_review).get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return " ".join(meaningful_words)  # returning a paragraph with stopwords and punctuations removed.


def main():
    file_name = "labeledTrainData.tsv"
    clean_reviews = []

    train_data_frame = read_data(file_name)
    clean_review = process_words(train_data_frame["review"][0])
    print clean_review
    print train_data_frame["review"].size

    for i in xrange(0, train_data_frame["review"].size):
        clean_reviews.append(process_words(train_data_frame["review"][i]))

    print len(clean_reviews)

    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None,stop_words=None, max_features=5000)
    train_data_features = vectorizer.fit_transform(clean_reviews)
    train_data_features = train_data_features.toarray() # converting to numpy array
    print train_data_features.shape

    vocab = vectorizer.get_feature_names() # this is simply the vocabulary - can be printed with the count of each word

    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_data_features, train_data_frame)

    test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
    print test.shape
    clean_test_reviews = []
    for i in xrange(0, len(test["review"])):
        clean_test_reviews.append(process_words(test["review"][i]))

    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    result = forest.predict(test_data_features)

    output_dataframe = pd.DataFrame(data={"id": test["id"], "sentiment": result})

    output_dataframe.to_csv("Bag_of_Words_model.csv", index=False, quoting=3)


main()
