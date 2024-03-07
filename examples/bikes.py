# Adapted from:
#
# https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html#sphx-glr-auto-examples-applications-plot-cyclical-feature-engineering-py


# +
from sklearn.datasets import fetch_openml

bike_sharing = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True)
df = bike_sharing.frame
y = df["count"]
y /= y.max()
X = df.drop("count", axis="columns")
X["rowid"] = X.index.values

# +
from skrub._pipe import Pipe

pipe = Pipe(X)
pipe

# +
pipe.get_skrubview_report(order_by="rowid")

# +
from sklearn.preprocessing import SplineTransformer
import numpy as np


def periodic_spline_transformer(period, n_splines=None, degree=3):
    if n_splines is None:
        n_splines = period
    n_knots = n_splines + 1
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True,
    )


# +
from skrub import selectors as s

pipe = pipe.chain(
    s.categorical().one_hot_encoder(sparse_output=False, handle_unknown="ignore"),
    s.cols("month").use(periodic_spline_transformer(12, n_splines=6)),
    s.cols("weekday").use(periodic_spline_transformer(7, n_splines=3)),
    s.cols("hour").use(periodic_spline_transformer(24, n_splines=12)),
    s.all().min_max_scaler(),
)
pipe

# +
pipe.get_skrubview_report(order_by="rowid")

# +
pipe.chain(
    (s.glob("hour_sp_*") | s.cols("workingday_True")).polynomial_features(
        degree=2, interaction_only=True, include_bias=False
    )
).get_skrubview_report(order_by="rowid")


# +
from skrub._pipe import choose

pipe = pipe.chain(
    (s.all() - "rowid").nystroem(
        kernel="poly", degree=2, n_components=choose(30, 300), random_state=0
    ),
)

pipe.get_skrubview_report(order_by="rowid")

# +
pipe = pipe.chain(
    s.cols("rowid").drop(), s.all().ridge_cv(alphas=np.logspace(-6, 6, 25))
)
# -


# Evaluation
# ----------

# +
from sklearn.model_selection import TimeSeriesSplit, cross_validate

ts_cv = TimeSeriesSplit(
    n_splits=5,
    gap=48,
    max_train_size=10000,
    test_size=1000,
)


def evaluate(model, X, y, cv, model_prop=None, model_step=None):
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error"],
        return_estimator=model_prop is not None,
    )
    if model_prop is not None:
        if model_step is not None:
            values = [
                getattr(m[model_step], model_prop) for m in cv_results["estimator"]
            ]
        else:
            values = [getattr(m, model_prop) for m in cv_results["estimator"]]
        print(f"Mean model.{model_prop} = {np.mean(values)}")
    mae = -cv_results["test_neg_mean_absolute_error"]
    rmse = -cv_results["test_neg_root_mean_squared_error"]
    print(
        f"Mean Absolute Error:     {mae.mean():.3f} +/- {mae.std():.3f}\n"
        f"Root Mean Squared Error: {rmse.mean():.3f} +/- {rmse.std():.3f}"
    )


# +
evaluate(pipe.grid_search, X, y, cv=ts_cv)
