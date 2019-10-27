import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

# To use this experimental feature, we need to explicitly ask for it:
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd
from sklearn.metrics import r2_score


def test_atp1d():
    df = pd.read_csv('/home/Kenny/Documents/atp1d.csv')
    target = df.loc[:, df.columns.str.startswith('LBL')]
    df.drop(target.columns, axis=1, inplace=True)
    df, target = df.to_numpy(), target.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.5, random_state=42)
    gb = HistGradientBoostingRegressor(
        verbose=1,
        random_state=42
    )
    gb.fit(X_train, y_train)
    y_preds = gb.predict_multi(X_test, np.shape(y_test)[1])
    r2 = r2_score(y_test, y_preds, multioutput='uniform_average')
    print(r2)

def test_edm():
    df = pd.read_csv('/home/Kenny/Documents/edm.csv')
    target = df.loc[:, ['DFlow', 'DGap']]
    df.drop(target.columns, axis=1, inplace=True)
    df, target = df.to_numpy(), target.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.5, random_state=42)
    gb = HistGradientBoostingRegressor(
        verbose=1,
        random_state=42
    )
    gb.fit(X_train, y_train)
    y_preds = gb.predict_multi(X_test, np.shape(y_test)[1])
    r2 = r2_score(y_test, y_preds, multioutput='uniform_average')
    print(r2)


def test_scm1d():
    df = pd.read_csv('/home/Kenny/Documents/scm1d.csv')
    target = df.loc[:, df.columns.str.contains('L')]
    df.drop(target.columns, axis=1, inplace=True)
    df, target = df.to_numpy(), target.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.5, random_state=42)
    gb = HistGradientBoostingRegressor(
        verbose=1,
        random_state=42
    )
    gb.fit(X_train, y_train)
    y_preds = gb.predict_multi(X_test, np.shape(y_test)[1])
    r2 = r2_score(y_test, y_preds, multioutput='uniform_average')
    print(r2)


def test_scm20d():
    df = pd.read_csv('/home/Kenny/Documents/scm20d.csv')
    target = df.loc[:, df.columns.str.contains('L')]
    df.drop(target.columns, axis=1, inplace=True)
    df, target = df.to_numpy(), target.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.5, random_state=42)
    gb = HistGradientBoostingRegressor(
        verbose=1,
        random_state=42
    )
    gb.fit(X_train, y_train)
    y_preds = gb.predict_multi(X_test, np.shape(y_test)[1])
    r2 = r2_score(y_test, y_preds, multioutput='uniform_average')
    print(r2)


def test_wq():
    df = pd.read_csv('/home/Kenny/Documents/water-quality.csv')
    target = df.loc[:, df.columns.str.startswith('x')]
    df.drop(target.columns, axis=1, inplace=True)
    df, target = df.to_numpy(), target.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.5, random_state=42)
    gb = HistGradientBoostingRegressor(
        verbose=1,
        random_state=42
    )
    gb.fit(X_train, y_train)
    y_preds = gb.predict_multi(X_test, np.shape(y_test)[1])
    r2 = r2_score(y_test, y_preds, multioutput='uniform_average')
    print(r2)