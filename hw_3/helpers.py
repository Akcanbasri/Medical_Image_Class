import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option("display.float_format", lambda x: "%.2f" % x)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def outlier_thresholds(df, col_name, q1=0.25, q3=0.75):
    """
    Calculate the lower and upper limits for outlier detection based on the interquartile range method.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the column to calculate the outlier thresholds for.
    col_name : str
        The name of the column to calculate the outlier thresholds for.
    q1 : float, optional
        The percentile value for the first quartile. Default is 0.25.
    q3 : float, optional
        The percentile value for the third quartile. Default is 0.75.

    Returns:
    --------
    low_limit : float
        The lower limit for outlier detection.
    up_limit : float
        The upper limit for outlier detection.
    """
    quartile1 = df[col_name].quantile(q1)
    quartile3 = df[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(df, variable):
    low_limit, up_limit = outlier_thresholds(df, variable)
    df.loc[(df[variable] < low_limit), variable] = low_limit
    df.loc[(df[variable] > up_limit), variable] = up_limit


def check_outlier(df, col_name):
    low_limit, up_limit = outlier_thresholds(df, col_name)
    if df[(df[col_name] > up_limit) | (df[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def grab_col_names(df, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        df: df
                Değişken isimleri alınmak istenilen df
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
    num_but_cat = [
        col
        for col in df.columns
        if df[col].nunique() < cat_th and df[col].dtypes != "O"
    ]
    cat_but_car = [
        col
        for col in df.columns
        if df[col].nunique() > car_th and df[col].dtypes == "O"
    ]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in df.columns if df[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {df.shape[0]}")
    print(f"Variables: {df.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car


def cat_summary(data_frame, colm_name, plot=False):
    """
    This function takes a pandas dataframe and a column name as input and returns a summary of the column's value counts
    and their ratios. If the plot parameter is set to True, it also displays a countplot of the column's values.

    Parameters:
    data_frame (pandas.DataFrame): The input dataframe
    colm_name (str): The name of the column to be summarized
    plot (bool): Whether to display a countplot of the column's values (default is False)

    Returns:
    None
    """
    print("###############################################")
    print(
        pd.DataFrame(
            {
                colm_name: data_frame[colm_name].value_counts(),
                "Ratio": 100 * data_frame[colm_name].value_counts() / len(data_frame),
            }
        )
    )

    if plot:
        sns.countplot(x=data_frame[colm_name], data=data_frame)
        plt.show(block=True)
    print("###############################################")


# cat cols için bool değişkenlerin int yapılması gerekiyor
# for i in cat_cols:
#     # bool değişkeni int yapma
#     if df[i].dtypes == "bool":
#         df[i] = df[i].astype(int)
#         cat_summary(df, i, plot=True)
#     else:
#         cat_summary(df, i, plot=True)


# num cols için özet istatistikler
def num_summary(dataframe, num_cols, plot=False):
    """
    This function takes a pandas dataframe and a list of numerical column names as input and returns a summary of the
    numerical columns including count, mean, standard deviation, minimum, 5th percentile, 10th percentile, 20th percentile,
    30th percentile, 40th percentile, 50th percentile (median), 60th percentile, 70th percentile, 80th percentile, 90th
    percentile, 95th percentile, 99th percentile and maximum. If plot is set to True, it also displays a histogram of
    the numerical columns.

    Parameters:
    dataframe (pandas.DataFrame): The pandas dataframe to be analyzed.
    num_cols (list): A list of numerical column names to be analyzed.
    plot (bool, optional): Whether to display a histogram of the numerical columns. Defaults to False.

    Returns:
    None
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[num_cols].describe(quantiles).T)

    if plot:
        dataframe[num_cols].hist(bins=20)
        plt.xlabel(num_cols)
        plt.title(num_cols)
        plt.show(block=True)


# num cols için çağırma
# for col in num_cols:
#     print("###############################################")
#     num_summary(df, col, plot=True)
#     print("###############################################")


# target analizi için özet istatistikler cat_cols
def target_summary_with_cat(dataframe, target, categorical_col):
    """
    This function takes in a dataframe, target column name and a categorical column name as input.
    It groups the dataframe by the categorical column and calculates the mean of the target column for each group.
    It then returns a pandas dataframe with the mean target values for each group.
    """
    print(
        pd.DataFrame(
            {"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}
        ),
        end="\n\n",
    )


# target analizi için özet istatistikler num_cols
def target_summary_with_num(dataframe, target, numerical_col):
    """
    This function groups the given dataframe by the target column and calculates the mean of the numerical column for each group.

    Parameters:
    dataframe (pandas.DataFrame): The input dataframe.
    target (str): The name of the target column.
    numerical_col (str): The name of the numerical column.

    Returns:
    None
    """
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n")


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[
        (dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)
    ].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def cat_summary(dataframe, col_name, plot=False):
    print(
        pd.DataFrame(
            {
                col_name: dataframe[col_name].value_counts(),
                "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe),
            }
        )
    )
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


def target_summary_with_cat(dataframe, target, categorical_col):
    print(
        pd.DataFrame(
            {"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}
        ),
        end="\n\n\n",
    )


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(
        np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool)
    )
    drop_list = [
        col
        for col in upper_triangle_matrix.columns
        if any(upper_triangle_matrix[col] > corr_th)
    ]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt

        sns.set(rc={"figure.figsize": (15, 15)})
        sns.heatmap(corr, cmap="RdBu", annot=True)
        plt.show(block=True)
    return drop_list


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(
            pd.DataFrame(
                {
                    "COUNT": dataframe[col].value_counts(),
                    "RATIO": dataframe[col].value_counts() / len(dataframe),
                    "TARGET_MEAN": dataframe.groupby(col)[target].mean(),
                }
            ),
            end="\n\n\n",
        )


def rare_encoder(dataframe, rare_perc, cat_cols):
    temp_df = dataframe.copy()
    rare_columns = [
        col
        for col in cat_cols
        if (temp_df[col].value_counts() / len(temp_df) < rare_perc).sum() > 1
    ]

    for col in rare_columns:
        tmp = temp_df[col].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[col] = np.where(temp_df[col].isin(rare_labels), "Rare", temp_df[col])

    return temp_df


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(
        dataframe, columns=categorical_cols, drop_first=drop_first
    )
    return dataframe
