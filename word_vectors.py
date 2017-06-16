import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
import logging
from gensim.models import word2vec, Word2Vec
from sklearn.cluster import KMeans
import numpy as np
from sklearn.ensemble import RandomForestClassifier


'''
Important discussions and references:
https://groups.google.com/forum/embed/#!topic/gensim/hlYgjqEVocw
https://www.crummy.com/software/BeautifulSoup/bs4/doc/
'''

'''
- grouping vectors in terms of clusters is called vector quantization. We can perform clustering using KMeans clustering.
- using more clusters with less density helps greatly to produce better results.
- Having a large "k" in kMeans clustering is very slow and hence we have decided to use it as a small value.
- word2vec actually creates word clusters. And our goal is to find the center of such word clusters.
- Conveniently, Word2Vec provides functions to load any pre-trained model that is output by Google's original C tool,
 so it's also possible to train a model in C and then import it into Python.
- Hierarchial softmax reduces the complexity from O(V) to O(logV). Hence it is also faster.
- The speed up in Hierarchial softmax comes during training. During training you only need to calculate the probability of one word
(Assuming CBOW model). You don't need to probability of every single word in your vocabulary.
'''

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

num_features = 300  # Word vector dimensionality
min_word_count = 40  # Minimum word count
num_workers = 4  # Number of threads to run in parallel
context = 10  # Context window size
downsampling = 1e-3  # Downsample setting for frequent words


def review_to_wordlist(review, remove_stopwords=False):
    review_text = BeautifulSoup(review).get_text()
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    return words


def review_to_sentences(review, tokenizer, remove_stopwords=False):
    raw_sentences = tokenizer.tokenize(review.strip().decode("utf8"))
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence,
                                                remove_stopwords))

    return sentences


def train_model(sentences):
    model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count,
                              window=context, sample=downsampling)
    model.init_sims(
        replace=True)  # init_sims is used when we do not plan to train the model further. Calling this will make the
    # model more memory-efficient

    model_name = "300features_40minwords_10context"
    model.save(model_name)


def cluster_words(word_vectors, model, num_clusters):
    logging.info("Clustering starting..")
    kmeans_clustering = KMeans(n_clusters=num_clusters)
    idx = kmeans_clustering.fit_predict(word_vectors)  # idx stores the cluster assignment for each word

    word_centroid_map = dict(zip(model.wv.index2word, idx))  # index2word is basically a list of words which are a part
    # of the vocabulary
    logging.info("word centroid map created..")
    return word_centroid_map


def create_bag_of_centroids(word_list, word_centroid_map):
    num_centroids = max(word_centroid_map.values()) + 1
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")

    for word in word_list:
        if word in word_centroid_map:
            idx = word_centroid_map[word]
            bag_of_centroids[idx] += 1

    return bag_of_centroids


def main():
    sentences = []
    train = pd.read_csv("labeledTrainData.tsv", header=0,
                        delimiter="\t", quoting=3)
    test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
    unlabeled_train = pd.read_csv("unlabeledTrainData.tsv", header=0,
                                  delimiter="\t", quoting=3)

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    for review in train["review"]:
        sentences += review_to_sentences(review, tokenizer)

    for review in unlabeled_train["review"]:
        sentences += review_to_sentences(review, tokenizer)

    # train_model(sentences)

    model = Word2Vec.load("300features_40minwords_10context")
    print model.doesnt_match("man woman child kitchen".split())
    print model.most_similar("man")
    print model.most_similar("queen")
    print model.most_similar("awful")

    print type(model.wv.syn0)
    print model.wv.syn0

    word_vectors = model.wv.syn0
    num_clusters = word_vectors.shape[0] / 5
    word_centroid_map = cluster_words(word_vectors, model, num_clusters=num_clusters)

    clean_train_reviews, clean_test_reviews = [], []
    for review in train["review"]:
        clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))

    for review in test["review"]:
        clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=True))

    train_centroids = np.zeros((train["review"].size, num_clusters), dtype="float32")
    test_centroids = np.zeros((test["review"].size, num_clusters), dtype="float32")

    counter = 0

    for review in clean_train_reviews:
        train_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
        counter += 1

    counter = 0
    for review in clean_test_reviews:
        test_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
        counter += 1

    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_centroids, train["sentiment"])
    result = forest.predict(test_centroids)
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("bag_of_centroids.csv", index=False, quoting=3)

main()
