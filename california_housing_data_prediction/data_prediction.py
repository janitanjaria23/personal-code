import urllib
import tarfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, LabelBinarizer

housing_path = "datasets/housing"
strat_train_set, strat_test_set = None, None


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
    plt.savefig("housing_data_histogram.png")


def create_scatter_plots(data_frame):
    data_frame.plot(kind="scatter", x="longitude", y="latitude")
    plt.savefig("housing_scatter_plot.png")
    data_frame.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)  # to show better distribution
    plt.savefig("housing_scatter_plot_with_alpha.png")

    data_frame.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=data_frame['population'],
                    label='population',
                    c='median_house_value',
                    cmap=plt.get_cmap("jet"), colorbar=True)
    plt.legend()
    plt.savefig("housing_scatter_plot_with_population.png")

    attributes = ["median_house_value", "median_income", "total_rooms",
                  "housing_median_age"]
    scatter_matrix(data_frame[attributes], figsize=(12, 8))


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


def get_correlation_info(data_frame):
    correlation_matrix = data_frame.corr()
    correlation_matrix['median_house_value'].sort_values(ascending=False)
    print correlation_matrix


def remove_non_numerical_attribute(data_frame, attribute_name):
    data_frame_num = data_frame.drop(attribute_name, axis=1)
    return data_frame_num


def convert_text_to_onehot_vectors(data_frame, attribute_name):
    data_frame_cat = data_frame[attribute_name]
    # encoder = LabelEncoder()
    # housing_cat_encoded = encoder.fit_transform(housing_cat)
    # print "Encoder Classes: ", encoder.classes_
    # encoder = OneHotEncoder()
    # housing_cat_onehot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
    encoder = LabelBinarizer()
    data_frame_cat_onehot = encoder.fit_transform(data_frame_cat)
    return data_frame_cat_onehot


def transform_data(data_frame, categorical_values_present=True):
    imputer = Imputer(strategy="median")
    data_frame_num = remove_non_numerical_attribute(data_frame=data_frame, attribute_name="ocean_proximity")
    imputer.fit(data_frame_num)
    print "Imputer Stats:", imputer.statistics_
    X = imputer.transform(data_frame_num)  # X is a plan numpy array containing transformed features
    # sometimes fit_transform() is optimized and runs much faster as compared to fit() + transform()
    data_frame_tr = pd.DataFrame(X, columns=data_frame_num.columns)
    if categorical_values_present:
        data_frame_cat_onehot = convert_text_to_onehot_vectors(data_frame, attribute_name="ocean_proximity")
        
    return data_frame_tr


def main():
    global strat_train_set, strat_test_set
    test_ratio = 0.2
    # fetch_housing_data()
    housing = load_housing_data()
    # explore_nature_of_df(data_frame=housing)
    create_histogram(data_frame=housing)
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

    # print housing['income_cat'].value_counts() / len(housing)

    for data_set in (strat_train_set, strat_test_set):
        data_set.drop(['income_cat'], axis=1, inplace=True)

    housing = strat_train_set.copy()
    # Scatterplot: https://www.wikiwand.com/en/Scatter_plot
    create_scatter_plots(data_frame=housing)
    housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
    plt.savefig("housing_correlation_with_income_plot.png")

    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]
    get_correlation_info(housing)

    housing = strat_train_set.drop("median_house_value",
                                   axis=1)  # .drop() creates a copy and does not affect the strat_train_set
    housing_labels = strat_train_set["median_house_value"].copy()


main()
