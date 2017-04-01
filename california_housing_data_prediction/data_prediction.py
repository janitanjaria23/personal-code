import urllib
import tarfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

housing_path = "datasets/housing"


def fetch_housing_data():
    data_path = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"
    housing_dataset_path = housing_path + "/housing.tgz"
    urllib.urlretrieve(data_path, housing_dataset_path)
    tar = tarfile.open(housing_path)
    tar.extractall()
    tar.close()


def load_housing_data(housing_path=housing_path):
    return pd.read_csv("housing.csv")


def split_train_test_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]


def create_histogram(data_frame):
    data_frame.hist(bins=50, figsize=(10, 15))  # can be done for individual attributes as well
    plt.show()


def explore_nature_of_df(data_frame):
    print data_frame.head()
    print data_frame.info()
    print data_frame['ocean_proximity'].value_counts()
    print data_frame.describe()


def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


def main():
    global strat_train_set, strat_test_set
    test_ratio = 0.2
    # fetch_housing_data()
    housing = load_housing_data()
    # explore_nature_of_df(data_frame=housing)
    # create_histogram(data_frame=housing)
    # train_set, test_set = split_train_test_data(data=housing,
    #                                             test_ratio=test_ratio)  # this can have the same set of data used multiple times as a test set - which will lead to false results
    # print "Size of training data: %d" % (len(train_set))
    # print "Size of testing data: %d" % (len(test_set))

    housing_with_id = housing.reset_index()
    print housing_with_id.info()
    # train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
    # print "Size of training data: %d" % (len(train_set))
    # print "Size of testing data: %d" % (len(test_set))

    train_set, test_set = train_test_split(housing, test_size=0.2,
                                           random_state=42)  # using inbuilt scikit learn data split function
    print "Size of training data: %d" % (len(train_set))
    print "Size of testing data: %d" % (len(test_set))

    housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
    housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing['income_cat']):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    print housing['income_cat'].value_counts() / len(housing)

    for data_set in (strat_train_set, strat_test_set):
        data_set.drop(['income_cat'], axis=1, inplace=True)

    housing = strat_train_set.copy()

    housing.plot(kind="scatter", x="longitude", y="latitude")

main()
