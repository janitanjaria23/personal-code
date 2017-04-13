from sklearn.datasets.mldata import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


def display_digit(some_digit):
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.savefig("digit_image.png")


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()
    plt.savefig("roc_curve.png")


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

    skfolds = StratifiedKFold(n_splits=3, random_state=42)

    for train_index, test_index in skfolds.split(X_train, y_train_5):
        clone_clf = clone(sgd_clf)  # clone creates a deep copy
        X_train_folds = X_train[train_index]
        y_train_folds = (y_train_5[train_index])
        X_test_fold = X_train[test_index]
        y_test_fold = (y_train_5[test_index])
        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)

        print n_correct / len(y_pred)

        print cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

        # the above result from cross validation is not good though the accuracy is high is because most of the times
        # the prediction is not5 and hence it is like the unknown problem

        y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5,
                                         cv=3)  # instead of evaluation scores, it returns the prediction results

        print confusion_matrix(y_train_5, y_train_pred)

        # a perfect classifier should just have true positives and true negatives i.e. just the main diagonal

        precision_score(y_train_5, y_pred)
        recall_score(y_train_5, y_train_pred)

        f1_score(y_train_5, y_pred)

        y_scores = sgd_clf.decision_function([X[36000]])
        print y_scores
        threshold = 0
        y_some_digit_pred = (y_scores > threshold)

        # by default SGD classifier has a threshold = 0. So the above is equivalent to .predict()

        y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

        fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
        plot_roc_curve(fpr, tpr)

        roc_auc_score(y_train_5, y_scores)  # for a perfect classifier the area under the ROC curve should be = 1

        forest_clf = RandomForestClassifier(random_state=42)
        y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                            method="predict_proba")
        y_scores_forest = y_probas_forest[:, 1]
        fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

        plt.plot(fpr, tpr, "b:", label="SGD")
        plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
        plt.legend(loc="bottom right")

        roc_auc_score(y_train_5, y_scores_forest)

        

main()
