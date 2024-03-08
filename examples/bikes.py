# Adapted from:
#
# https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html#sphx-glr-auto-examples-applications-plot-cyclical-feature-engineering-py


# +
from joblib.numpy_pickle import load_temporary_memmap
from sklearn.datasets import fetch_openml

bike_sharing = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True)
df = bike_sharing.frame
y = df["count"]
y /= y.max()
X = df.drop("count", axis="columns")
X["rowid"] = X.index.values

# +
from skrub import selectors as s
from skrub._pipe import Pipe

pipe = Pipe(X)
pipe
# -

# We can get a html preview of the pipelines' output. Maybe that should be the
# pipeline's `_html_repr_`. OTOH it can take a few seconds to generate the
# report when there are hundreds of columns. Also we can have some parameters
# to set, such as the `order_by="rowid"` which causes the report to show line
# plots rather than histograms.

# +
pipe.get_skrubview_report(order_by="rowid")
# -

# Note that we can select columns by clicking their checkbox and then get a
# list of column names in the bar at the top by selecting 'selected columns' in
# the dropdown. That can save us some typing when specifying transformations
# for specific columns.
#
# Rather than a random sample we may want to zoom in on the first few rows.
# This gives a better overview of the columns that vary quickly. Here we could
# have more methods such as passing a slice etc.

# +
pipe.get_skrubview_report(order_by="rowid", sampling_method="head")
# -

# Let's check the categorical variables when sampling all rows, to see all the unique values.
# As we see below that helps us discover the rare category 'heavy rain' so we
# know our encoding of that column will need to handle that.

# +
pipe.chain((~s.categorical()).drop()).get_skrubview_report(n=float("inf"))
# -

# Ok enough procrastinating, let's add some steps to the pipeline.

# +
from sklearn.preprocessing import SplineTransformer
import numpy as np


# Same transformer as in the original example


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
pipe = pipe.chain(
    s.categorical().one_hot_encoder(sparse_output=False, handle_unknown="ignore"),
    s.cols("month").use(periodic_spline_transformer(12, n_splines=6)),
    s.cols("weekday").use(periodic_spline_transformer(7, n_splines=3)),
    s.cols("hour").use(periodic_spline_transformer(24, n_splines=12)),
    (s.all() - "rowid").min_max_scaler(),
)
pipe

# +
pipe.get_skrubview_report(order_by="rowid")
# -

# Here also, if we want a better look at the fast-varying features we can
# sample the first rows instead.

# +
pipe.get_skrubview_report(order_by="rowid", sampling_method="head")
# -

# We make polynomial interactions between the hour spline features and the
# column that tells us if it's a working day. Here we rely on the column names
# chosen by the one-hot and spline transformers to grab the columns we want. If
# we wanted something more robust we could use the `.rename_columns()` method
# of the steps to insert tags on which we would rely, eg
#
# ```
# pipe.chain(
#     s.cols("hour")
#     .use(periodic_spline_transformer(24, n_splines=12))
#     .rename_columns("<hour spline>{}")
# )
# ```
#
# and later
#
# ```
# pipe.chain(s.glob('<hour spline>*').some_estimator())
# ```

# +
pipe = pipe.chain(
    (s.glob("hour_sp_*") | s.cols("workingday_True")).polynomial_features(
        degree=2, interaction_only=True, include_bias=False
    ),
    # get rid of the hour spline * hour spline interactions
    s.glob("hour_sp_* hour_sp_*").drop(),
)
pipe.get_skrubview_report(order_by="rowid", sampling_method="head")
# -

# Let's not use the polynomial features after all

# +
_ = pipe.pop()

# +
from skrub._pipe import choose

pipe = pipe.chain(
    (s.all() - "rowid").nystroem(
        kernel="poly", degree=2, n_components=choose(10, 300), random_state=0
    ),
)
# choose(10, 300) is just to reduce the length of the default preview but then
# the gridsearch will pick 300.

pipe.get_skrubview_report(order_by="rowid", sampling_method="head")

# +
pipe = pipe.chain(
    s.cols("rowid").drop(), s.all().ridge_cv(alphas=np.logspace(-6, 6, 25))
)
# We can have a final look at our pipeline:
print(pipe.get_pipeline_description())
# -


# Evaluation
# ----------
# as in the original example

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
evaluate(pipe.get_grid_search(), X, y, cv=ts_cv)
