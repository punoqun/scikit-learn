import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

# To use this experimental feature, we need to explicitly ask for it:
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.utils import shuffle
import pandas as pd


X_classification, y_classification = make_classification(random_state=0)
X_regression, y_regression = make_regression(n_samples=1000,random_state=0, n_targets=1)


@pytest.mark.parametrize('GradientBoosting, X, y', [
    (HistGradientBoostingClassifier, X_classification, y_classification),
    (HistGradientBoostingRegressor, X_regression, y_regression)
])
@pytest.mark.parametrize(
    'params, err_msg',
    [({'loss': 'blah'}, 'Loss blah is not supported for'),
     ({'learning_rate': 0}, 'learning_rate=0 must be strictly positive'),
     ({'learning_rate': -1}, 'learning_rate=-1 must be strictly positive'),
     ({'max_iter': 0}, 'max_iter=0 must not be smaller than 1'),
     ({'max_leaf_nodes': 0}, 'max_leaf_nodes=0 should not be smaller than 2'),
     ({'max_leaf_nodes': 1}, 'max_leaf_nodes=1 should not be smaller than 2'),
     ({'max_depth': 0}, 'max_depth=0 should not be smaller than 2'),
     ({'max_depth': 1}, 'max_depth=1 should not be smaller than 2'),
     ({'min_samples_leaf': 0}, 'min_samples_leaf=0 should not be smaller'),
     ({'l2_regularization': -1}, 'l2_regularization=-1 must be positive'),
     ({'max_bins': 1}, 'max_bins=1 should be no smaller than 2 and no larger'),
     ({'max_bins': 257}, 'max_bins=257 should be no smaller than 2 and no'),
     ({'n_iter_no_change': -1}, 'n_iter_no_change=-1 must be positive'),
     ({'validation_fraction': -1}, 'validation_fraction=-1 must be strictly'),
     ({'validation_fraction': 0}, 'validation_fraction=0 must be strictly'),
     ({'tol': -1}, 'tol=-1 must not be smaller than 0')]
)
def test_init_parameters_validation(GradientBoosting, X, y, params, err_msg):

    with pytest.raises(ValueError, match=err_msg):
        GradientBoosting(**params).fit(X, y)


def test_invalid_classification_loss():
    binary_clf = HistGradientBoostingClassifier(loss="binary_crossentropy")
    err_msg = ("loss='binary_crossentropy' is not defined for multiclass "
               "classification with n_classes=3, use "
               "loss='categorical_crossentropy' instead")
    with pytest.raises(ValueError, match=err_msg):
        binary_clf.fit(np.zeros(shape=(3, 2)), np.arange(3))


@pytest.mark.parametrize(
    'scoring, validation_fraction, n_iter_no_change, tol', [
        ('neg_mean_squared_error', .1, 5, 1e-7),  # use scorer
        ('neg_mean_squared_error', None, 5, 1e-1),  # use scorer on train data
        (None, .1, 5, 1e-7),  # same with default scorer
        (None, None, 5, 1e-1),
        ('loss', .1, 5, 1e-7),  # use loss
        ('loss', None, 5, 1e-1),  # use loss on training data
        (None, None, None, None),  # no early stopping
        ])
def test_early_stopping_regression(scoring, validation_fraction,
                                   n_iter_no_change, tol):

    max_iter = 200

    X, y = make_regression(n_samples=50, random_state=0, n_targets=2)

    gb = HistGradientBoostingRegressor(
        verbose=1,  # just for coverage
        min_samples_leaf=5,  # easier to overfit fast
        scoring=scoring,
        tol=tol,
        validation_fraction=validation_fraction,
        max_iter=max_iter,
        # n_iter_no_change=n_iter_no_change,
        random_state=0
    )
    gb.fit(X, y)
    # gb.predict(X)
    preds = gb.predict_multi(X, np.shape(y)[1])
    if n_iter_no_change is not None:
        assert n_iter_no_change <= gb.n_iter_ < max_iter
    else:
        assert gb.n_iter_ == max_iter


@pytest.mark.parametrize('data', (
    make_classification(n_samples=30, random_state=0),
    make_classification(n_samples=30, n_classes=3, n_clusters_per_class=1,
                        random_state=0)
))
@pytest.mark.parametrize(
    'scoring, validation_fraction, n_iter_no_change, tol', [
        ('accuracy', .1, 5, 1e-7),  # use scorer
        ('accuracy', None, 5, 1e-1),  # use scorer on training data
        (None, .1, 5, 1e-7),  # same with default scorerscor
        (None, None, 5, 1e-1),
        ('loss', .1, 5, 1e-7),  # use loss
        ('loss', None, 5, 1e-1),  # use loss on training data
        (None, None, None, None),  # no early stopping
        ])
def test_early_stopping_classification(data, scoring, validation_fraction,
                                       n_iter_no_change, tol):

    max_iter = 50

    X, y = data

    gb = HistGradientBoostingClassifier(
        verbose=1,  # just for coverage
        min_samples_leaf=5,  # easier to overfit fast
        scoring=scoring,
        tol=tol,
        validation_fraction=validation_fraction,
        max_iter=max_iter,
        n_iter_no_change=n_iter_no_change,
        random_state=0
    )
    gb.fit(X, y)

    if n_iter_no_change is not None:
        assert n_iter_no_change <= gb.n_iter_ < max_iter
    else:
        assert gb.n_iter_ == max_iter


@pytest.mark.parametrize(
    'scores, n_iter_no_change, tol, stopping',
    [
        ([], 1, 0.001, False),  # not enough iterations
        ([1, 1, 1], 5, 0.001, False),  # not enough iterations
        ([1, 1, 1, 1, 1], 5, 0.001, False),  # not enough iterations
        ([1, 2, 3, 4, 5, 6], 5, 0.001, False),  # significant improvement
        ([1, 2, 3, 4, 5, 6], 5, 0., False),  # significant improvement
        ([1, 2, 3, 4, 5, 6], 5, 0.999, False),  # significant improvement
        ([1, 2, 3, 4, 5, 6], 5, 5 - 1e-5, False),  # significant improvement
        ([1] * 6, 5, 0., True),  # no significant improvement
        ([1] * 6, 5, 0.001, True),  # no significant improvement
        ([1] * 6, 5, 5, True),  # no significant improvement
    ]
)
def test_should_stop(scores, n_iter_no_change, tol, stopping):

    gbdt = HistGradientBoostingClassifier(
        n_iter_no_change=n_iter_no_change, tol=tol
    )
    assert gbdt._should_stop(scores) == stopping


def test_binning_train_validation_are_separated():
    # Make sure training and validation data are binned separately.
    # See issue 13926

    rng = np.random.RandomState(0)
    validation_fraction = .2
    gb = HistGradientBoostingClassifier(
        n_iter_no_change=5,
        validation_fraction=validation_fraction,
        random_state=rng
    )
    gb.fit(X_classification, y_classification)
    mapper_training_data = gb.bin_mapper_

    # Note that since the data is small there is no subsampling and the
    # random_state doesn't matter
    mapper_whole_data = _BinMapper(random_state=0)
    mapper_whole_data.fit(X_classification)

    n_samples = X_classification.shape[0]
    assert np.all(mapper_training_data.actual_n_bins_ ==
                  int((1 - validation_fraction) * n_samples))
    assert np.all(mapper_training_data.actual_n_bins_ !=
                  mapper_whole_data.actual_n_bins_)


@pytest.mark.parametrize('data', [
    make_classification(random_state=0, n_classes=2),
    make_classification(random_state=0, n_classes=3, n_informative=3)
], ids=['binary_crossentropy', 'categorical_crossentropy'])
def test_zero_division_hessians(data):
    # non regression test for issue #14018
    # make sure we avoid zero division errors when computing the leaves values.

    # If the learning rate is too high, the raw predictions are bad and will
    # saturate the softmax (or sigmoid in binary classif). This leads to
    # probabilities being exactly 0 or 1, gradients being constant, and
    # hessians being zero.
    X, y = data
    gb = HistGradientBoostingClassifier(learning_rate=100, max_iter=10)
    gb.fit(X, y)


def test_small_trainset():
    # Make sure that the small trainset is stratified and has the expected
    # length (10k samples)
    n_samples = 20000
    original_distrib = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples).reshape(n_samples, 1)
    y = [[class_] * int(prop * n_samples) for (class_, prop)
         in original_distrib.items()]
    y = shuffle(np.concatenate(y))
    gb = HistGradientBoostingClassifier()

    # Compute the small training set
    X_small, y_small = gb._get_small_trainset(X, y, seed=42)

    # Compute the class distribution in the small training set
    unique, counts = np.unique(y_small, return_counts=True)
    small_distrib = {class_: count / 10000 for (class_, count)
                     in zip(unique, counts)}

    # Test that the small training set has the expected length
    assert X_small.shape[0] == 10000
    assert y_small.shape[0] == 10000

    # Test that the class distributions in the whole dataset and in the small
    # training set are identical
    assert small_distrib == pytest.approx(original_distrib)


def test_infinite_values():

    X = np.array([-np.inf, 0, 1, np.inf]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1])

    gbdt = HistGradientBoostingRegressor(min_samples_leaf=1)
    gbdt.fit(X, y)
    np.testing.assert_allclose(gbdt.predict(X), y, atol=1e-4)
