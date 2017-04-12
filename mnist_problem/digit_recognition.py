from sklearn.datasets.mldata import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier


def display_digit(some_digit):
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.savefig("digit_image.png")


def main():
    mnist = fetch_mldata("MNIST original", data_home="/Users/anjaria.janit/scikit_learn_data/")
    X, y = mnist['data'], mnist['target']
    print 'Shape of X: ', X.shape
    print 'Shape of y: ', y.shape
    display_digit(X[36000])

    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    shuffle_index = np.random.permutation(60000)

    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)

    sgd_clf = SGDClassifier(random_state=42)  # random_state helps to create reproducible results
    sgd_clf.fit(X_train, y_train_5)
    print sgd_clf.predict([X[36000]])
    

main()
