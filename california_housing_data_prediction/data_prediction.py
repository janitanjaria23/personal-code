import urllib
import tarfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, LabelBinarizer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor


housing_path = "datasets/housing"
strat_train_set, strat_test_set = None, None

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


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

    return data_frame_cat_onehot


def use_linear_regression(housing_prepared, housing_labels):
    linear_reg = LinearRegression()
    linear_reg.fit(housing_prepared, housing_labels)
    housing_predictions = linear_reg.predict(housing_prepared)
    linear_mse = mean_squared_error(housing_labels, housing_predictions)
    linear_rmse = np.sqrt(linear_mse)
    print "Linear Regression RMSE: ", linear_rmse
    return linear_reg


def use_decision_tree_regressor(housing_prepared, housing_labels):
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(housing_prepared, housing_labels)
    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    print "Decision Tree RMSE: ", tree_rmse
    return tree_reg


def display_scores(model_name, housing_prepared, housing_labels):
    scores = cross_val_score(model_name, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
    print scores
    rmse_scores = np.sqrt(-scores)
    print "Scores: ", rmse_scores
    print "Mean Scores: ", rmse_scores.mean()
    print "Std. Deviation Scores: ", rmse_scores.std()


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

    num_pipeline = Pipeline([("imputer", Imputer(strategy="median")), ("attribs_adder", CombinedAttributesAdder()),
                             ("std_scaler", StandardScaler()), ])
    # housing_num = remove_non_numerical_attribute(data_frame=housing, attribute_name="ocean_proximity")
    housing_num = housing.drop("ocean_proximity", axis=1)
    # housing_num_tr = num_pipeline.fit_transform(housing_num)

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', LabelBinarizer()),
    ])

    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

    housing_prepared = full_pipeline.fit_transform(housing)
    # print housing_prepared
    print housing_prepared.shape

    linear_reg = use_linear_regression(housing_prepared, housing_labels)
    tree_reg = use_decision_tree_regressor(housing_prepared, housing_labels)
    display_scores(linear_reg, housing_prepared, housing_labels)
    display_scores(tree_reg, housing_prepared, housing_labels)


main()
