import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


insuranca = pd.read_csv("data/insurance.csv")
df = insuranca.copy()
# call data

df.head()
# df.head() call first 5 rows of data

df.tail(3)
# df.columns call columns name

df.info()
# df.info() call data type and null values

df.dtypes
# df.dtypes call data type

df.describe().T
# df.describe() call statistical values

# df.describe(include="0").T
# df.describe(include="0") call categorical values
# but this code is not working because there is no categorical values

df.shape
# df.shape call data shape

df.isna()
# data.isna() call null values r

df.isnull().any()
# df.isnull().any() call null values return true or false

df.isnull().sum()
# df.isnull().sum() call null values return sum of null values

df = df.dropna(axis="columns")
# df = df.dropna(axis='columns') drop null values but we dont have null values

df["sex"].value_counts()
# count of sex values

print()
#####################
# Data Visualization#
#####################

sns.countplot(x="sex", data=df)
plt.title("Sec Count")
plt.show()
# countplot

sns.histplot(df.charges, bins=50)
plt.title("Age Distribution")
plt.show()
# histogram

sns.histplot(df.age, bins=50)
plt.title("Age Distribution")
plt.show()
# histogram

sns.pairplot(df, hue="sex")
plt.show()
# pairplot show us correlation between columns

print()
#####################
# Correlation Matrix#
#####################

corr = df.corr()
sns.heatmap(corr, annot=True, cmap="RdYlGn")
plt.title("Correlation Matrix", fontsize=15, fontweight="bold")
plt.show()
# heatmap

corr[abs(corr["charges"]) > 0.59].index
# our data correlation is good because of this we dont need to drop any columns
