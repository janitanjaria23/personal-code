import pandas as pd
import numpy as np
import pylab as pyl
from sklearn.ensemble import RandomForestClassifier


def transform_data(df):
    df['Age'].dropna().hist(bins=16, range=(0, 80), alpha=0.5)
    # pyl.show()

    df['Gender'] = 4

    df['Gender'] = df['Sex'].map(lambda x: x[0].upper())

    df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # print df.head(3)

    median_ages = np.zeros((2, 3))
    # print median_ages

    for i in range(0, 2):
        for j in range(0, 3):
            median_ages[i, j] = df[(df['Gender'] == i) & (df['Pclass'] == j + 1)]['Age'].dropna().median()

    # print median_ages

    df['AgeFill'] = df['Age']

    # print df.head(3)

    # print df[df['Age'].isnull()][['Gender', 'Pclass', 'Age', 'AgeFill']].head(10)

    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j + 1), 'AgeFill'] = median_ages[i, j]

    # print df[df['Age'].isnull()][['Gender', 'Pclass', 'Age', 'AgeFill']].head(10)

    df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

    df['FamilySize'] = df['SibSp'] + df['Parch']

    df['Age*Class'] = df['AgeFill'] * df['Pclass']
    # df['Embarked_New'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2, 'Nan': 3}).astype(int)

    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)

    df = df.drop(['Age'], axis=1)

    df = df.dropna()

    # print df.head(3)

    return df

df = pd.read_csv('/var/data/practice_data/titanic_train_data.csv', header=0)
df = transform_data(df)

print df.head(3)

# print type(df)
# print df.dtypes
# print df.info()
# print df.describe()
# print df['Age'][0:10]
# print df['Cabin']
#
# print df[df['Age'] > 60][['Sex', 'Pclass', 'Age', 'Survived']]
# print df[df['Age'].isnull()][['Sex', 'Pclass', 'Age', 'Survived']]


train_data = df.values


forest = RandomForestClassifier(n_estimators=100)

print train_data[0::, 1::]

forest = forest.fit(train_data[0::, 1::], train_data[0::, 0])
