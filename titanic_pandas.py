import pandas as pd
import numpy as np
import pylab as pyl

df = pd.read_csv('/var/data/practice_data/titanic_train_data.csv', header=0)
print df.head(3)
print type(df)
print df.dtypes
print df.info()
print df.describe()
print df['Age'][0:10]
print df['Cabin']

print df[df['Age'] > 60][['Sex', 'Pclass', 'Age', 'Survived']]
print df[df['Age'].isnull()][['Sex', 'Pclass', 'Age', 'Survived']]

df['Age'].dropna().hist(bins=16, range=(0, 80), alpha=0.5)
pyl.show()