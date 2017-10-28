import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt


def dropna_rows(df, thres_frac=1.0, verbose=True):
    """ Drop rows (dataframe indexes) in which the total number of missing values exceeds the threshold thres_frac.
    For example: thres_frac=1 => drops rows where all elements are na
                 thres_frac=0.5 => drops rows where half of the elements are na
    Args:
        df (pd.DataFrame): input dataframe
        thres_frac (float): threshold of min number of na elements in a row as a fraction of total number of elements
    Returns:
        df (pd.DataFrame): updated dataframe
        row_index (pd.Series): boolean vector where True indicates the dropped rows
        rows_dropped (list): list of dropped rows (indexes)
    """
    assert not df.empty, "df is empty."
    assert (thres_frac >= 0) and (thres_frac <= 1), "thres_frac must be in the range [0, 1]."
    df = df.copy()

    thres_elements = int(df.shape[1] * thres_frac)
    row_index = df.isnull().sum(axis=1) >= thres_elements
    rows_dropped = []

    if np.sum(row_index):
        rows_dropped = list(df.loc[row_index, :].index)
        df = df.loc[~row_index, :]

    if verbose:
        print("\n{} rows out of {} were dropped based on the number of missing values in a row (thres_frac={}).".format(
            len(rows_dropped), len(row_index), thres_frac))

    return df, row_index, rows_dropped


def dropna_cols(df, thres_frac=1.0, verbose=True):
    """ Drop cols (dataframe columns) in which the total number of missing values exceeds the threshold thres_frac.
    For example: thres_frac=1 => drops cols if all elements are na
                 thres_frac=0.5 => drops cols where half of the elements are na
    Args:
        df (pd.DataFrame): input dataframe
        thres_frac (int): threshold of min number of na elements in a col as a fraction of total number of elements
    Returns:
        df (pd.DataFrame): updated dataframe
        cols_dropped (list): list of dropped columns
    """
    assert not df.empty, "df is empty."
    assert (thres_frac >= 0) and (thres_frac <= 1), "thres_frac must be in the range [0, 1]."
    df = df.copy()

    thres_elements = int(df.shape[0] * thres_frac)
    col_index = df.isnull().sum(axis=0) >= thres_elements
    cols_dropped = []

    if np.sum(col_index):
        cols_dropped = list(df.loc[:, col_index].columns)
        df = df.loc[:, ~col_index]

    if verbose:
        print("\n{} cols out of {} were dropped based on the number of missing values in a col (thres_frac={}).".format(
            len(cols_dropped), len(col_index), thres_frac))

    return df, cols_dropped


def split_discrete_and_continuous(df, thres_discrete=2, verbose=True):
    """ Split discrete cols (i.e. cols that contain less than a specified number of unique values; excluding na values)
    and continuous cols.
    Args:
        df (pd.DataFrame): input dataframe
        thres_discrete (int): max number of unique values in a col (excluding na values) to consider the col as discrete
    Returns:
        dfc (pd.DataFrame): dataframe containing continuous values
        dfd (pd.DataFrame): dataframe containing discrete values
    """
    assert not df.empty, "df is empty."
    if thres_discrete < 2:
        if verbose:
            print("\nDidn't split dataframe into continuous and discrete features. Split requirement: "
                  "thres_discrete >= 2.")
        return df, pd.DataFrame()
    df = df.copy()

    # Find columns which contain <=thres_discrete unique values (excluding na values)
    col_index = [True if len(df.loc[:, c].dropna().unique()) <= int(thres_discrete) else False for c in df.columns]
    col_index = np.array(col_index)

    dfc = df.loc[:, ~col_index]  # df with continuous columns
    dfd = df.loc[:, col_index]  # df with discrete columns

    if verbose:
        print("\nSplit dataset into 2 dataframes:"
              "\ndfc: contains {} continous features"
              "\ndfc: contains {} discrete features".format(dfc.shape[1], dfd.shape[1]))

    return dfc, dfd


def preproc_discrete(df, impute_value=None, onehot=True):
    """ Preprocess discrete cols.
    Impute strategies --> https://www.quora.com/How-can-I-deal-with-missing-values-in-a-predictive-model
    Note! encoding len(df.columns) takes long time (check for alternative approaches)
    Args:
        df (pd.DataFrame): input dataframe
        impute_value (str): a value to use for imputing the na values
        onehot (bool): whether to one-hot the discrete cols
    Returns:
        df (pd.DataFrame): updated dataframe
    """
    from sklearn.preprocessing import LabelEncoder
    if df.empty:
        return df
    df = df.copy()

    # Encode the "discrete" values into integers (using LabelEncoder)
    for c in df.columns:
        na_index = df.loc[:, c].isnull().values
        # Impute if there are na values
        if np.sum(na_index):
            if impute_value is None:
                # If impute_value=None, then impute with a value that is smaller then the min in this col
                df.loc[na_index, c] = np.nanmin(df.loc[:, c].unique()) - 1
            else:
                df.loc[na_index, c] = impute_value

        # Encode values (encoding len(df.columns) takes long time)
        df.loc[:, c] = LabelEncoder().fit_transform(df.loc[:, c])

    # One hot encode the encoded cols
    if onehot:
        df = pd.get_dummies(df, prefix=df.columns, columns=df.columns)
    # or --> df = pd.concat([pd.get_dummies(df[c], prefix=c) for c in df.columns], axis=1)

    # TODO: Drop cols based on conditions applicable to discrete values (e.g., chi-squared)

    return df


def preproc_continuous(df, scaling='std', impute_value=None, drop_strategy='corrcoef', thres_corr=0.95,
                       create_iom='no'):
    """ Preprocess continuous cols.
    Impute strategies --> https://www.quora.com/How-can-I-deal-with-missing-values-in-a-predictive-model
    1. Impute missing values
    2. Drop cols based on conditions
    3. Scale cols
    Args:
        df (pd.DataFrame): input dataframe
        scaling (str): scaling strategy
        impute_value (str): a value to use for imputing the na values ('mean', 'median', 'mode', if nothing passed then
                            imputes with 0)
        drop_strategy (str): drop strategy
        thres_corr (float): drop cols whose correlations with other cols exceeds this threshold
        create_iom (bool): whether to create an indicator of missingness
    Returns:
        df (pd.DataFrame): updated dataframe
    """
    from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
    if df.empty:
        return df
    df = df.copy()

    df1 = df.loc[:, df.isnull().sum(axis=0) == 0].copy()  # extract cols which don't contain na values
    df2 = df.loc[:, df.isnull().sum(axis=0) > 0].copy()  # extract cols which contain na values

    # Impute continuous cols and create new binary cols which are Indicators Of Missingness (IOM)
    if create_iom.lower() == 'yes':
        df_binr = pd.DataFrame(index=df2.index)  # IOM dataframe
        for c in df2.columns:
            # Create new col in IOM dataframe
            na_index = df2.loc[:, c].isnull().values
            df_binr[c + '_na'] = na_index.astype(np.int32)
            # Impute
            if impute_value == 'mean':
                df2.loc[na_index, c] = df2.loc[:, c].mean(skipna=True)
            elif impute_value == 'median':
                df2.loc[na_index, c] = df2.loc[:, c].median(skipna=True)
            elif impute_value == 'mode':
                # mode() may return multiple values (see pandas doc)
                df2.loc[na_index, c] = np.unique(df2.loc[:, c].mode())[0]
            else:
                df2.loc[na_index, c] = 0

    elif create_iom.lower() == 'no':
        if impute_value == 'mean':
            df2 = df2.apply(lambda x: x.where(~np.isnan(x), x[~x.isnull().values].mean()))
        elif impute_value == 'median':
            df2 = df2.apply(lambda x: x.where(~np.isnan(x), x[~x.isnull().values].median())).copy()
        elif impute_value == 'mode':
            # mode() may return multiple values (see pandas doc)
            df2 = df2.apply(lambda x: x.where(~np.isnan(x), np.unique(x[~x.isnull().values].mode())[0])).copy()
        else:
            df2 = df2.apply(lambda x: x.where(~np.isnan(x), 0)).copy()

    # Concatenate continuous dataframes (excluding IOM dataframe)
    assert (len(np.unique(df1.index == df2.index)) == 1) & (np.unique(df1.index == df2.index) == True), \
        "Index mismatch when concatenating df1 and df2."
    df = pd.concat([df1, df2], axis=1, ignore_index=False)

    # Drop cols based on conditions (process the combined df)
    if drop_strategy == 'corrcoef':
        df, _ = drop_high_correlation_cols(df, thres_corr=thres_corr, verbose=True)

    # Get the indexes and columns of the dataframe
    df_cols = df.columns
    df_idx = df.index

    # Scale only continuous columns (i.e. don't scale the IOM dataframe)
    if scaling == 'maxabs':
        df = MaxAbsScaler().fit_transform(df)
    elif scaling == 'minmax':
        df = MinMaxScaler().fit_transform(df)
    elif scaling == 'std':
        df = StandardScaler().fit_transform(df)
    else:
        assert False, "the scaling `{}` is not supported".format(scaling)

    # The scaling methods return numpy arrays (not dataframes) thus we recreate the dataframe
    df = pd.DataFrame(df, columns=df_cols, index=df_idx)

    # Concatenate continuous and binary (indicators of missingness) dataframes
    if create_iom.lower() == 'yes':
        df = pd.concat([df, df_binr], axis=1, ignore_index=False)

    return df


def drop_high_correlation_cols(df, thres_corr=0.95, verbose=True):
    """ Remove cols whose correlations with other cols exceed thres_corr.
    Note: cols with var=0 (yield corr=nan) and na values are not processed here; should be processed before/after.
    For example: thres_corr=1 => drops col if there is a perfect correlation with other col
                 thres_corr=0.5 => drops col if there is 0.5 correlation with other col
    Args:
        df (pd.DataFrame): input dataframe (df should not contain any missing values)
        thres_corr (float): min correlation value to drop the cols
    Returns:
        df (pd.DataFrame): updated dataframe
        cols_dropped (list): cols/features dropped
    """
    assert (thres_corr >= 0) & (thres_corr <= 1), "thres_corr must be in the range [0, 1]."
    df = df.copy()

    col_len = len(df.columns)

    # Ignore cols which contain missing values and where var = 0
    idx_tmp = (df.var(axis=0) == 0).values | (df.isnull().sum(axis=0) > 0).values
    df_tmp = df.loc[:, idx_tmp].copy()
    df = df.loc[:, ~idx_tmp].copy()  # dataframe to be processed in this function

    # compute pearson correlation coeff (https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
    # mat = df.values
    # mean_vec = mat.mean(axis=0, keepdims=True)
    # std_vec = mat.std(axis=0, keepdims=True)
    # mat_corr = np.dot(mat.T - mean_vec.T, mat - mean_vec)
    # tmp1 = np.sqrt((mat.T - mean_vec.T) ** 2)
    # tmp2 = np.sqrt((mat - mean_vec) ** 2)
    # mat_std = np.dot(tmp1, tmp2)
    # CC = mat_corr/mat_std

    # corr can be computed using --> df.corr(method='pearson').abs() --> this is much slower than using just numpy!
    cc = np.corrcoef(df.values.T)
    corr = pd.DataFrame(cc, index=df.columns, columns=df.columns).abs()

    # Iteratively update the correlation matrix
    # Iterate as long as there are correlations above thres_corr (excluding autocorrelation values, i.e. on diagonal)
    cols_dropped = []
    while (~np.eye(len(corr), dtype=np.bool) * (corr >= thres_corr)).any().sum():
        # print('corr matrix: shape={}'.format(corr.shape))
        thres_mask = ~np.eye(len(corr), dtype=np.bool) * (corr >= thres_corr)  # mask relevant indexes --> where corr>=thres_corr
        corr_count = thres_mask.sum(axis=1)  # count occurance of relevant indexes for each col --> how many times a col has corr>thres_corr
        col_index = corr_count[corr_count == corr_count.max()].index  # get indexes of most relevant cols --> 'most relevant' = most occurances
        corr_weight = (thres_mask * corr).sum(axis=1)  # assign weight (sum of relevant corr values) to each col

        # Among the most relevant cols (defined by col_index),
        # choose the col with the max corr weight to be dropped
        col_drop = corr_weight[col_index].argmax()
        cols_dropped.append(col_drop)

        # Remove the col from the corr matrix
        corr = corr.loc[~(corr.index == col_drop), ~(corr.columns == col_drop)]

    # Update the original dataset
    # corr.index contains columns names that are left after removing high correlations columns
    df = df.loc[:, corr.index]

    # Concatenate processed df and df_tmp
    df = pd.concat([df, df_tmp], axis=1, ignore_index=False)

    if verbose:
        print("\n{} cols were dropped based on high cross-correlation between cols (thres_corr={}).".format(
              col_len-len(df.columns), thres_corr))

    return df, cols_dropped


def drop_low_variance_cols(df, thres_var=0, skipna=True, verbose=True):
    """ Drop cols in which the variance is lower than thres_var.
    Args:
        df (pd.DataFrame): input dataframe
        thres_var (int): threshold variance to drop the cols
    Returns:
        df (pd.DataFrame): updated dataframe
    """
    assert not df.empty, "df is empty."
    df = df.copy()

    index = df.var(axis=0, skipna=skipna) <= thres_var
    df = df.loc[:, ~index]
    if verbose:
        print('\n{} cols out of {} were dropped based on col variance (thres_var={}).'.format(
              np.sum(index), len(index), thres_var))

    return df, index


def store_results(df):
    """ Stores experimental conditions (input args in command line) and results (accuracy/R^2/loss) in csv file.
    Loads the csv file if exists, otherwise creates a new one.
    """
    # TODO


def create_test_data():
    """ Create a dataset in order to test the different preprocessing funcations available in this script. """
    df1 = pd.DataFrame({'c00': [np.nan, 1] + list(np.zeros((10,)) * np.nan),  # all na values
                        'c01': [np.nan, np.nan] + [1] + list(np.zeros((9,)) * np.nan),  # extremely many na values
                        'c02': [np.nan, np.nan] + [1, 1, 1] + list(np.zeros((7,)) * np.nan),  # many na values
                        'c03': [np.nan, np.nan] + list(np.ones((4,))) + list(np.ones((4,)) * 0) + [np.nan, np.nan],
                        # discrete with na
                        'c04': [np.nan, np.nan] + list(np.random.randn(10, )),  # continuous variable
                        'c05': [np.nan, np.nan] + list(np.random.randn(8, )) + [np.nan, np.nan],
                        # continuous variable with na
                        'c06': [np.nan, np.nan] + list(np.ones((1,)) * 0) + list(
                            np.ones((9,)) * 0)})  # zero or low variance

    tmp = np.random.randn(10)
    df2 = pd.DataFrame({'c07': [np.nan, np.nan] + list(np.ones((10,)) * 2),
                        'c08': [np.nan, np.nan] + list(np.ones((10,)) * 3),
                        'c09': [np.nan, np.nan] + list(np.arange(10)),
                        'c10': [np.nan, np.nan] + list(2 * np.arange(10) + 1),  # scaled (positive) version of c9
                        'c11': [np.nan, np.nan] + list(-2 * np.arange(10) + 1),  # scaled (negative) version of c9
                        'c12': [np.nan, np.nan] + list(2 * np.arange(10) * np.random.rand(10) * 0.03 + 1),
                        'c13': [np.nan, np.nan] + list(tmp),
                        'c14': [np.nan, np.nan] + list(2 * tmp - 1)})  # scaled (positive) version of c13

    df = pd.concat([df1, df2], axis=1, ignore_index=False)
    return df


def test_preproc(df, scaling='std', thres_frac_rows=1, thres_frac_cols=1, thres_var=0, thres_corr=1,
                 thres_discrete=0, impute_value='mean', create_iom=False, verbose=True):
    """ Testbench for data_preproc.py """
    #df = df.dropna(axis=0, how='all')
    #df = df.dropna(axis=1, how='all')

    df, row_index, _ = dropna_rows(df, thres_frac=thres_frac_rows, verbose=True)
    df, _ = dropna_cols(df, thres_frac=thres_frac_cols, verbose=True)

    dfc, dfd = split_discrete_and_continuous(df, thres_discrete=thres_discrete)

    dfc = preproc_continuous(dfc, scaling=scaling, impute_value=impute_value, thres_corr=thres_corr, create_iom=create_iom)
    if not dfd.empty:
        dfd = preproc_discrete(dfd)  # takes a few minutes to process

    if (not dfc.empty) and (not dfd.empty):
        assert len(np.unique(dfd.index == dfc.index)) == 1, "Index mismatch when concatenating discrete and continuous dataframes."
        dff = pd.concat([dfc, dfd], axis=1)
    elif not dfd.empty:
        dff = dfd
    elif not dfc.empty:
        dff = dfc

    dff, cols_dropped = drop_low_variance_cols(dff, thres_var=thres_var, skipna=True, verbose=True)
    return dff


# df = create_test_data()
# dff = test_preproc(df, verbose=True)
# dff = test_preproc(df, scaling='std', thres_frac_rows=0.9, thres_frac_cols=0.80, thres_var=0, thres_corr=0.95,
#                    thres_discrete=2, create_iom='yes', verbose=True)

