import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split, cross_validate

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.width", 500)

diabets = pd.read_csv("data/diabetes.csv")
df = diabets.copy()

df.shape

df.head()

df.tail(3)

df.info()
# df.info() call data type and null values

df.dtypes
# df.dtypes call data type

df.describe().T
# df.describe() call statistical values

df.isna()
# data.isna() call null values

df.isnull().any()
# df.isnull().any() call null values return true or false

df.isnull().sum()
# df.isnull().sum() call null values return sum of null values

df = df.dropna(axis="columns")
# df = df.dropna(axis='columns') drop null values but we dont have null values

df = df.drop("id", axis=1)

df.head()

num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
# Correlation Matrix and Heatmap
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="viridis")
plt.show()


corr_matrix = corr.abs()
# Select upper triangle of correlation matrix and return index of columns bigger than 0.90
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
drop_list = [col for col in upper.columns if any(upper[col] > 0.90)]

corr_matrix[drop_list]

df = df.drop(drop_list, axis=1)

df.head()

from sklearn.preprocessing import LabelEncoder

# label encoding


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [
    col
    for col in df.columns
    if df[col].dtype not in [int, float] and df[col].nunique() == 2
]

le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

df.head()
# 1 = M and 0 = B

sns.countplot(x="diagnosis", data=df)
plt.title("Diagnosis")
plt.show()
# data visualization


sns.histplot(df["radius_mean"])
plt.title("Radius Mean")
plt.show()

sns.histplot(df["texture_mean"])
plt.title("Texture Mean")
plt.show()

y = df["diagnosis"]
X = df.drop(["diagnosis"], axis=1)
# Holdout method
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=17
)

log_model = LogisticRegression().fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))

"""
For X_train
accuracy: 0.95
precision: 0.94
recall: 0.90
f1-score: 0.92
roc_auc_score: 0.9880353417597251
***************************
For X_test
accuracy: 0.95
precision: 0.92
recall: 0.92
f1-score: 0.92
roc_auc_score: 0.974376731301939
"""


RocCurveDisplay.from_estimator(log_model, X_test, y_test)
plt.title("ROC Curve")
plt.plot([0, 1], [0, 1], "r--")
plt.show()
# ROC Curve

roc_auc_score(y_test, y_prob)


X = df.drop(["diagnosis"], axis=1)
y = df["diagnosis"]
# Model validation with cross validation
log_model = LogisticRegression().fit(X, y)

cv_results = cross_validate(
    log_model, X, y, cv=10, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
)

cv_results["test_accuracy"].mean()
#0.9331766917293234
cv_results["test_precision"].mean()
# 0.9342157477025899
cv_results["test_recall"].mean()
# 0.8878787878787879
cv_results["test_f1"].mean()
# 0.9067462903663162
cv_results["test_roc_auc"].mean()
# 0.9841606541606541

"""
For X_train
accuracy: 0.95  / doğru olarak sınıflandırılanların oranı
precision: 0.94 / pozitif olarak tahmin edilenlerin gerçekten pozitif olma oranı (TP/(TP+FP)
recall: 0.90 / gerçekten pozitif olanların pozitif olarak tahmin edilme oranı (TP/(TP+FN)
f1-score: 0.92 / precision ve recall harmonik ortalaması
roc_auc_score: 0.9880353417597251 / roc eğrisi altında kalan alan
***************************
For X_test
accuracy: 0.95
precision: 0.92
recall: 0.92
f1-score: 0.92
roc_auc_score: 0.974376731301939
***************************
For Cross Validation
accuracy: 0.93
precision: 0.93
recall: 0.89
f1-score: 0.91
roc_auc_score: 0.98

***************************
maybe overfitting
***************************
"""

# confusion matriz yaz