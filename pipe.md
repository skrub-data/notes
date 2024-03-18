---
breaks: false
---

# A high-level interface for building a dataframe transformation pipeline

See also:
- Vision / step forward for skrub https://hackmd.io/@GaelVaroquaux/ryzYaLO6T

The goal is to have a more high-level and interactive interface for building a scikit-learn pipeline.
In particular it should offer:

- previews of a sample of the data transformed with the current pipeline.
- a way to very easily add steps that rely on one of the most commonly-used estimators such as `Ridge`, `HistGradientBoostingRegressor`.
- a way to add steps that rely on any scikit-learn compatible estimator
- a way to specify ranges of hyperparameters (for tuning) as the pipeline is being constructed.
- a way to perform cross-validation and hyperparameter search, eg by obtaining a scikit-learn GridSearchCV or Pipeline and using scikit-learn cross-validation tools, once we satisfied with the pipeline we built.

The prototype used for the examples here (which will be updated as we make decisions) is in [this branch](https://github.com/jeromedockes/skrub/tree/pipeline).

here is some toy data:

```python
import pandas as pd

df = pd.DataFrame(
    {
        "A": list(range(1, 6)),
        "B": "one one two two two".split(),
        "C": ["01/02/1998", "10/03/2027", "11/02/2012", "23/04/1999", "01/01/1901"],
        "D": [n + 0.5 for n in range(5)],
        "E": [n + 5.2 for n in range(5)],
    }
).convert_dtypes()

df
```
<!-- output -->
```
   A    B           C    D    E
0  1  one  01/02/1998  0.5  5.2
1  2  one  10/03/2027  1.5  6.2
2  3  two  11/02/2012  2.5  7.2
3  4  two  23/04/1999  3.5  8.2
4  5  two  01/01/1901  4.5  9.2
```

# Applying some transformations

The pipeline is instantiated with a dataset so we can get the previews.

__Note:__ don't pay attention to the skrub imports for now; anything we decide to put in the public API will be importable directly from `skrub`.

```python
from skrub._pipe import Pipe
from skrub import selectors as s

pipe = Pipe(df)
pipe
```
<!-- output -->
```
<Pipe: 0 transformations>
Sample of transformed data:
   A    B           C    D    E
2  3  two  11/02/2012  2.5  7.2
0  1  one  01/02/1998  0.5  5.2
1  2  one  10/03/2027  1.5  6.2
3  4  two  23/04/1999  3.5  8.2
4  5  two  01/01/1901  4.5  9.2
```

Roughly 2 APIs for adding steps are being considered; other suggestions welcome.

ATM it seems Option 1 is the main candidate and option 2 may or may not be added as a more "advanced" interface in the future.

## Option 1: with `Pipe.use()`

In this option, the pipeline has a `use` method (can also be named `apply`, for example, if that's not too confusing with pandas' apply).
We pass it the transformer to use, and optional configuration such as the columns on which to use it or a name for the step as kwargs.

```python
from skrub._to_datetime import ToDatetime
from skrub._datetime_encoder import EncodeDatetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from skrub._to_numeric import ToNumeric

p = (
    pipe
    .use(ToDatetime(), cols="C")
    .use(EncodeDatetime(), cols=s.any_date(), name="encode-dt")
    .use(OneHotEncoder(sparse_output=False), cols=s.string())
    .use(Ridge())
)
p
```
<!-- output -->
```
<Pipe: 3 transformations + predictor>
Steps:
0: to_datetime, 1: encode-dt, 2: one_hot_encoder, 3: ridge
Sample of transformed data:
   A  C_year  C_month  C_day  C_total_seconds    D    E  B_one  B_two
0  3  2012.0      2.0   11.0     1328918400.0  2.5  7.2    0.0    1.0
1  1  1998.0      2.0    1.0      886291200.0  0.5  5.2    1.0    0.0
2  2  2027.0      3.0   10.0     1804636800.0  1.5  6.2    1.0    0.0
3  4  1999.0      4.0   23.0      924825600.0  3.5  8.2    0.0    1.0
4  5  1901.0      1.0    1.0    -2177452800.0  4.5  9.2    0.0    1.0
```

By default the preview is a random sample, we can also see the first few rows:

```python
p.sample(sampling_method="head")
```
<!-- output -->
```
   A  C_year  C_month  C_day  C_total_seconds    D    E  B_one  B_two
0  1  1998.0      2.0    1.0      886291200.0  0.5  5.2    1.0    0.0
1  2  2027.0      3.0   10.0     1804636800.0  1.5  6.2    1.0    0.0
2  3  2012.0      2.0   11.0     1328918400.0  2.5  7.2    0.0    1.0
3  4  1999.0      4.0   23.0      924825600.0  3.5  8.2    0.0    1.0
4  5  1901.0      1.0    1.0    -2177452800.0  4.5  9.2    0.0    1.0
```


Notes:

- the `name` parameter sets the step name in the scikit-learn pipeline.
  It could be something more explicit like `step_name`.
- The preview shows the transformation part of the pipeline only, ie stops before the final predictor if there is one.


<details>
<summary>Implicitly selecting columns for which a transformer applies.</summary>
A transformer can reject columns for which it doesn't apply, in which case they are passed through.
For example whether a string column contains dates can only be discovered when trying to parse them.
Instead of `pipe.use(ToDatetime(), cols="C")`, if we didn't know in advance which columns contain dates, we could have written `pipe.use(ToDatetime())` and the result would be the same.
We can have a "strict" mode (which can be the default) where that would result in an error and we would be forced to specify `pipe.use(ToDatetime(), cols="C")`.
See [#877](https://github.com/skrub-data/skrub/pull/877) for more discussion.
</details>

<br/>

We can then extract a scikit-learn Pipeline that we can cross-validate etc.

```python
p.get_pipeline()
```
<!-- output -->
```
NamedParamPipeline(steps=[('to_datetime',
                           OnEachColumn(cols=['C'], transformer=ToDatetime())),
                          ('encode-dt',
                           OnEachColumn(cols=any_date(), transformer=EncodeDatetime())),
                          ('one_hot_encoder',
                           OnColumnSelection(cols=string(), transformer=OneHotEncoder(sparse_output=False))),
                          ('ridge', Ridge())])
```

This is a regular scikit-learn `Pipeline`, with `fit` and `transform` or `predict` methods.
We can also see a more human-readable summary of the steps.

```python
print(p.get_pipeline_description())
```
<!-- output -->
```
to_datetime:
    cols: ['C']
    estimator: ToDatetime()
encode-dt:
    cols: any_date()
    estimator: EncodeDatetime()
one_hot_encoder:
    cols: string()
    estimator: OneHotEncoder(sparse_output=False)
ridge:
    cols: all()
    estimator: Ridge()

```

If the transformation fails we see at which step it failed and the input data for the failing step:

```python
from sklearn.preprocessing import StandardScaler

(pipe
 .use(ToDatetime())
 .use(StandardScaler())
 .use(Ridge()))
```
<!-- output -->
```
<Pipe: 2 transformations + predictor>
Steps:
0: to_datetime, 1: standard_scaler, 2: ridge
Transformation failed at step 'standard_scaler'.
Input data for this step:
   A    B          C    D    E
2  3  two 2012-02-11  2.5  7.2
0  1  one 1998-02-01  0.5  5.2
1  2  one 2027-03-10  1.5  6.2
3  4  two 1999-04-23  3.5  8.2
4  5  two 1901-01-01  4.5  9.2
Error message:
    ValueError: could not convert string to float: 'two'
Note:
    Use `.sample()` to trigger the error again and see the full traceback.
    You can remove steps from the pipeline with `.pop()`.
```

`.sample()` doesn't catch the exception so it can be inspected.

We can also ask to see only the part of the output that was created by the last step:

```python
(pipe
 .use(ToDatetime(), cols="C")
 .use(EncodeDatetime(), cols=s.any_date())
 .sample(last_step_only=True))

```
<!-- output -->
```
   C_year  C_month  C_day  C_total_seconds
2  2012.0      2.0   11.0     1328918400.0
0  1998.0      2.0    1.0      886291200.0
1  2027.0      3.0   10.0     1804636800.0
3  1999.0      4.0   23.0      924825600.0
4  1901.0      1.0    1.0    -2177452800.0
```

TODO: give that parameter a better name or add other methods instead, eg `.sample_last_step()`

## Option 2: with `Selector.to_datetime()`, `Selector.use()`

We can also have the `use` method directly on the selectors, some methods for commonly used estimators.
This last point avoids having to import estimators and provides tab-completion on their name in interactive shells.
Another difference is configuration like the step name is added with an additional method call rather than additional kwargs.

```python
pipe.chain(
    s.cols("C").to_datetime(),
    s.any_date().encode_datetime().name("encode-dt"),
    s.string().one_hot_encoder(sparse_output=False),
    s.all().ridge(),
)
```
<!-- output -->
```
<Pipe: 3 transformations + predictor>
Steps:
0: to_datetime, 1: encode-dt, 2: one_hot_encoder, 3: ridge
Sample of transformed data:
   A  C_year  C_month  C_day  C_total_seconds    D    E  B_one  B_two
0  3  2012.0      2.0   11.0     1328918400.0  2.5  7.2    0.0    1.0
1  1  1998.0      2.0    1.0      886291200.0  0.5  5.2    1.0    0.0
2  2  2027.0      3.0   10.0     1804636800.0  1.5  6.2    1.0    0.0
3  4  1999.0      4.0   23.0      924825600.0  3.5  8.2    0.0    1.0
4  5  1901.0      1.0    1.0    -2177452800.0  4.5  9.2    0.0    1.0
```

(Instead of `chain` it could be `apply`, `use`, `transform`, `with_steps`, ...)

We can also pass directly an estimator (in which case the column selection is `s.all()`), and the selectors have a `.use()` method for using estimators that haven't been registered as methods.
So this is equivalent to the above:

```python
pipe.chain(
    s.cols("C").to_datetime(),
    s.any_date().encode_datetime().name("encode-dt"),
    s.string().use(OneHotEncoder(sparse_output=False)),
    Ridge(),
)
```
<!-- output -->
```
<Pipe: 3 transformations + predictor>
Steps:
0: to_datetime, 1: encode-dt, 2: one_hot_encoder, 3: ridge
Sample of transformed data:
   A  C_year  C_month  C_day  C_total_seconds    D    E  B_one  B_two
0  3  2012.0      2.0   11.0     1328918400.0  2.5  7.2    0.0    1.0
1  1  1998.0      2.0    1.0      886291200.0  0.5  5.2    1.0    0.0
2  2  2027.0      3.0   10.0     1804636800.0  1.5  6.2    1.0    0.0
3  4  1999.0      4.0   23.0      924825600.0  3.5  8.2    0.0    1.0
4  5  1901.0      1.0    1.0    -2177452800.0  4.5  9.2    0.0    1.0
```

## Discarded options

<details>

### Option3: with `Pipe.cols().to_datetime()`, `Pipe.cols().use()`

The third option adds a method `.cols` (or maybe `.on_cols`) to the pipeline to which we pass the selector.
That returns an object that is used to configure the next step

```
(
    pipe.cols("C")
    .to_datetime()
    .cols(s.any_date())
    .encode_datetime()
    .cols(s.string())
    .one_hot_encoder(sparse_output=False)
    .cols(s.all())
    .ridge()
)
```
<!-- output -->
```
<Pipe: 3 transformations + Ridge>
Steps:
0: to_datetime, 1: encode_datetime, 2: one_hot_encoder, 3: ridge
Sample of transformed data:
   A  C_year  C_month  C_day  C_total_seconds    D    E  B_one  B_two
0  3  2012.0      2.0   11.0     1328918400.0  2.5  7.2    0.0    1.0
1  1  1998.0      2.0    1.0      886291200.0  0.5  5.2    1.0    0.0
2  2  2027.0      3.0   10.0     1804636800.0  1.5  6.2    1.0    0.0
3  4  1999.0      4.0   23.0      924825600.0  3.5  8.2    0.0    1.0
4  5  1901.0      1.0    1.0    -2177452800.0  4.5  9.2    0.0    1.0
```

Notes:

Methods that add an estimator (eg `encode_datetime()`) have to return the `Pipe` object itself, so it's not clear where we should provide configuration such as the step name.
That may not be very important, as ATM I don't see anything else than the step name to configure (there could be a `param_grid` but the other way of specifying it described later seems better), and the step name may not be that important.

We could also say that there is a `.name()` method on the `Pipe` itself that implicitly applies to the last step.

We cannot pass an estimator directly, but the result of `cols` has a `use()` (or `apply()`, or ...) method:

```
(
    pipe.cols("C")
    .use(ToDatetime())
    .cols(s.any_date())
    .use(EncodeDatetime())
    .cols(s.string())
    .use(OneHotEncoder(sparse_output=False))
    .cols(s.all())
    .use(Ridge())
)
```
<!-- output -->
```
<Pipe: 3 transformations + Ridge>
Steps:
0: to_datetime, 1: encode_datetime, 2: one_hot_encoder, 3: ridge
Sample of transformed data:
   A  C_year  C_month  C_day  C_total_seconds    D    E  B_one  B_two
0  3  2012.0      2.0   11.0     1328918400.0  2.5  7.2    0.0    1.0
1  1  1998.0      2.0    1.0      886291200.0  0.5  5.2    1.0    0.0
2  2  2027.0      3.0   10.0     1804636800.0  1.5  6.2    1.0    0.0
3  4  1999.0      4.0   23.0      924825600.0  3.5  8.2    0.0    1.0
4  5  1901.0      1.0    1.0    -2177452800.0  4.5  9.2    0.0    1.0
```

As the `.cols()` looks like we are indexing the data it may be a bit surprising if someone expects the result of the transformation on just those columns to be returned:

```
p = pipe.cols(["A", "B"]).one_hot_encoder(sparse_output=False)
p
```
<!-- output -->
```
<Pipe: 1 transformations>
Steps:
0: one_hot_encoder
Sample of transformed data:
            C    D    E  A_1.0  A_2.0  A_3.0  A_4.0  A_5.0  B_one  B_two
0  11/02/2012  2.5  7.2    0.0    0.0    1.0    0.0    0.0    0.0    1.0
1  01/02/1998  0.5  5.2    1.0    0.0    0.0    0.0    0.0    1.0    0.0
2  10/03/2027  1.5  6.2    0.0    1.0    0.0    0.0    0.0    1.0    0.0
3  23/04/1999  3.5  8.2    0.0    0.0    0.0    1.0    0.0    0.0    1.0
4  01/01/1901  4.5  9.2    0.0    0.0    0.0    0.0    1.0    0.0    1.0
```

A user could be surprised to see "C", "D", "E", and "F" in the output above.

### Option 4

Having the estimator methods directly on the `Pipe` rather than on `pipe.cols`

```
(
    pipe
    .to_datetime().on_cols("C")
    .encode_datetime().on_cols(s.any_date()).name("encode-dt")
    .one_hot_encoder(sparse_output=False).on_cols(s.string())
    .ridge()
)
```

Note that this one would _require_ having methods on `Pipe` such as `on_cols` that implicitly apply to the last step.

</details>


# Choosing hyperparameters

It is important to be able to tune hyperparameters, and thus to provide a parameter grid to scikit-learn's `GridSearchCV`, `RandomizedSearchCV` or successive halving.

Manually specifying a large list of dicts all at once is not very easy because:
- the hyperparameters are not next to the corresponding estimator
- we have to refer to the estimators by their step names

Instead, we can have a `choose()` function that wraps the hyperparameter and pass it directly to the estimator.

The `Choice` object returned by `choose` has a `.name()` method, which we can use to give a more human-friendly name to that hyperparameter choice.
That could be used when displaying cross-validation results.
Otherwise we always have the usual `step_name__param_name` grid-search name.


## Example with `Pipe.use`

```python
from skrub._pipe import choose

p = (
    pipe
    .use(ToDatetime(), cols="C")
    .use(
        EncodeDatetime(resolution=choose("month", "day").name("time res")),
        cols=s.any_date(),
    )
    .use(OneHotEncoder(sparse_output=False), cols=s.string())
    .use(Ridge(alpha=choose(1.0, 10.0).name("α")))
)
p
```
<!-- output -->
```
<Pipe: 3 transformations + predictor>
Steps:
0: to_datetime, 1: encode_datetime, 2: one_hot_encoder, 3: ridge
Sample of transformed data:
   A  C_year  C_month  C_total_seconds    D    E  B_one  B_two
0  3  2012.0      2.0     1328918400.0  2.5  7.2    0.0    1.0
1  1  1998.0      2.0      886291200.0  0.5  5.2    1.0    0.0
2  2  2027.0      3.0     1804636800.0  1.5  6.2    1.0    0.0
3  4  1999.0      4.0      924825600.0  3.5  8.2    0.0    1.0
4  5  1901.0      1.0    -2177452800.0  4.5  9.2    0.0    1.0
```

We can see a summary of the hyperparameter grid:

```python
print(p.get_param_grid_description())
```
<!-- output -->
```
- 'time res':
      - 'month'
      - 'day'
  'α':
      - 1.0
      - 10.0

```

(and of the steps)

```python
print(p.get_pipeline_description())
```
<!-- output -->
```
to_datetime:
    cols: ['C']
    estimator: ToDatetime()
encode_datetime:
    cols: any_date()
    estimator: EncodeDatetime(resolution=choose('month', 'day').name('time res'))
one_hot_encoder:
    cols: string()
    estimator: OneHotEncoder(sparse_output=False)
ridge:
    cols: all()
    estimator: Ridge(alpha=choose(1.0, 10.0).name('α'))

```

And we can obtain a scikit-learn `GridSearchCV` or `RandomizedSearchCV` that we can use to tune hyperparameters.
This is not yet in the prototybe but we should also have methods (or a parameter) to get a successive halving object as well.

```python
p.get_grid_search()
```
<!-- output -->
```
GridSearchCV(estimator=NamedParamPipeline(steps=[('to_datetime',
                                                  OnEachColumn(cols=['C'], transformer=ToDatetime())),
                                                 ('encode_datetime',
                                                  OnEachColumn(cols=any_date(), transformer=EncodeDatetime(resolution='month'))),
                                                 ('one_hot_encoder',
                                                  OnColumnSelection(cols=string(), transformer=OneHotEncoder(sparse_output=False))),
                                                 ('ridge', Ridge())]),
             param_grid=[{'encode_datetime__transformer__resolution': choose('month', 'day').name('time res'),
                          'ridge__alpha': choose(1.0, 10.0).name('α')}])
```


<details>
<summary>hyperparameter choice with the alternative APIs</summary>

## `Selector.use` (option 2)

```python
p = pipe.chain(
    s.cols("C").to_datetime(),
    s.any_date().encode_datetime(resolution=choose("month", "day").name("time res")),
    s.string().one_hot_encoder(sparse_output=False),
    s.all().ridge(alpha=choose(1.0, 10.0).name("α")),
)

print(p.get_param_grid_description())
```
<!-- output -->
```
- 'time res':
      - 'month'
      - 'day'
  'α':
      - 1.0
      - 10.0

```

</details>


# Choices in nested estimators

Using `choose` for sub-estimators or their hyperparameters works as expected.

```python
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LogisticRegression, RidgeClassifier

regressor = BaggingRegressor(
    choose(
        RidgeClassifier(alpha=choose(1.0, 10.0).name("α")),
        LogisticRegression(C=choose(0.1, 1.0).name("C")),
    ).name("bagged")
)
p = (
    pipe.use(ToDatetime(), cols="C")
    .use(
        EncodeDatetime(resolution=choose("month", "day").name("time res")),
        cols=s.any_date(),
    )
    .use(OneHotEncoder(sparse_output=False), cols=s.string())
    .use(regressor)
)
p
```
<!-- output -->
```
<Pipe: 3 transformations + predictor>
Steps:
0: to_datetime, 1: encode_datetime, 2: one_hot_encoder, 3: bagging_regressor
Sample of transformed data:
   A  C_year  C_month  C_total_seconds    D    E  B_one  B_two
0  3  2012.0      2.0     1328918400.0  2.5  7.2    0.0    1.0
1  1  1998.0      2.0      886291200.0  0.5  5.2    1.0    0.0
2  2  2027.0      3.0     1804636800.0  1.5  6.2    1.0    0.0
3  4  1999.0      4.0      924825600.0  3.5  8.2    0.0    1.0
4  5  1901.0      1.0    -2177452800.0  4.5  9.2    0.0    1.0
```

```python
print(p.get_pipeline_description())
```
<!-- output -->
```
to_datetime:
    cols: ['C']
    estimator: ToDatetime()
encode_datetime:
    cols: any_date()
    estimator: EncodeDatetime(resolution=choose('month', 'day').name('time res'))
one_hot_encoder:
    cols: string()
    estimator: OneHotEncoder(sparse_output=False)
bagging_regressor:
    cols: all()
    estimator: BaggingRegressor(estimator=choose(RidgeClassifier(alpha=choose(1.0, 10.0).name('α')), LogisticRegression(C=choose(0.1, 1.0).name('C'))).name('bagged'))

```

```python
print(p.get_param_grid_description())
```
<!-- output -->
```
- 'time res':
      - 'month'
      - 'day'
  'bagged': RidgeClassifier(alpha=<α>)
  'α':
      - 1.0
      - 10.0
- 'time res':
      - 'month'
      - 'day'
  'bagged': LogisticRegression(C=<C>)
  'C':
      - 0.1
      - 1.0

```

# Naming options

If we want to give a name to individual choices we can pass keyword arguments to `choose`.
This can be useful to get more human-readable descriptions of pipelines and parameters.
The example above can be adapted:

```python
regressor = BaggingRegressor(
    choose(
        ridge=RidgeClassifier(alpha=choose(1.0, 10.0).name("α")),
        logistic=LogisticRegression(C=choose(0.1, 1.0).name("C")),
    ).name("bagged")
)
p = pipe.use(regressor)
print(p.get_param_grid_description())
```
<!-- output -->
```
- 'bagged': 'ridge'
  'α':
      - 1.0
      - 10.0
- 'bagged': 'logistic'
  'C':
      - 0.1
      - 1.0

```

(note if we want to use names that are not valid python identifiers we can always use the dict unpacking syntax `choose(**{'my name': 10})`).

# Choosing among several estimators

We may also want to choose among several estimators, ie have a choice for the whole step.
We can pass a `Choice` to `use`: `pipe.use(choose(RidgeClassifier(), LogisticRegression()))`.
`optional` is a shorthand for choosing between a step and passthrough.
We also have `choose_int` and `choose_float` to get int or floats within a range in a linear or log scale, possibly discretized.


```python
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from skrub._pipe import choose_float, optional

p = pipe.use(ToDatetime(), cols="C")
p = p.use(
    EncodeDatetime(resolution=choose("month", "day").name("time res")),
    cols=s.any_date(),
)
p = p.use(
    choose(
        one_hot=OneHotEncoder(sparse_output=False),
        ordinal=OrdinalEncoder(),
    ),
    cols=s.string(),
    name="cat-encoder",
)
p = p.use(optional(StandardScaler()))
p = p.use(
    choose(
        ridge=RidgeClassifier(alpha=choose_float(0.01, 100.0, log=True).name("α")),
        logistic=LogisticRegression(C=choose_float(0.01, 100.0, log=True).name("C")),
    ),
    name="classifier",
)
p
```
<!-- output -->
```
<Pipe: 4 transformations + predictor>
Steps:
0: to_datetime, 1: encode_datetime, 2: cat-encoder, 3: standard_scaler, 4: classifier
Sample of transformed data:
          A    C_year   C_month  ...         E     B_one     B_two
0  0.000000  0.553258 -0.392232  ...  0.000000 -0.816497  0.816497
1 -1.414214  0.238396 -0.392232  ... -1.414214  1.224745 -1.224745
2 -0.707107  0.890610  0.588348  ... -0.707107  1.224745 -1.224745
3  0.707107  0.260886  1.568929  ...  0.707107 -0.816497  0.816497
4  1.414214 -1.943149 -1.372813  ...  1.414214 -0.816497  0.816497

[5 rows x 8 columns]
```

```python
print(p.get_pipeline_description())
```
<!-- output -->
```
to_datetime:
    cols: ['C']
    estimator: ToDatetime()
encode_datetime:
    cols: any_date()
    estimator: EncodeDatetime(resolution=choose('month', 'day').name('time res'))
cat-encoder:
    cols: string()
    choose estimator from:
        - one_hot = OneHotEncoder(sparse_output=False)
        - ordinal = OrdinalEncoder()
standard_scaler:
    OPTIONAL STEP
    cols: all()
    estimator: StandardScaler()
classifier:
    cols: all()
    choose estimator from:
        - ridge = RidgeClassifier(alpha=choose_float(0.01, 100.0, log=True).name('α'))
        - logistic = LogisticRegression(C=choose_float(0.01, 100.0, log=True).name('C'))

```

```python
print(p.get_param_grid_description())
```
<!-- output -->
```
- 'time res':
      - 'month'
      - 'day'
  'cat-encoder':
      - 'one_hot'
      - 'ordinal'
  'standard_scaler':
      - 'true'
      - 'false'
  'classifier': 'ridge'
  'α': choose_float(0.01, 100.0, log=True)
- 'time res':
      - 'month'
      - 'day'
  'cat-encoder':
      - 'one_hot'
      - 'ordinal'
  'standard_scaler':
      - 'true'
      - 'false'
  'classifier': 'logistic'
  'C': choose_float(0.01, 100.0, log=True)

```


<details>
<summary>With the alternative APIs</summary>

## `Selector.use` (option 2)

```python
p = pipe.chain(
    s.cols("C").to_datetime(),
    s.any_date().encode_datetime(resolution=choose("month", "day").name("time res")),
    s.string().use(
        choose(
            one_hot=OneHotEncoder(sparse_output=False),
            ordinal=OrdinalEncoder())
        ).name("encoder"),
    choose(
        ridge=RidgeClassifier(alpha=choose(1.0, 10.0).name("α")),
        logistic=LogisticRegression(C=choose(0.1, 1.0).name("C")),
    ).name("classifier"),
)

print(p.get_param_grid_description())
```
<!-- output -->
```
- 'time res':
      - 'month'
      - 'day'
  'encoder':
      - 'one_hot'
      - 'ordinal'
  'classifier': 'ridge'
  'α':
      - 1.0
      - 10.0
- 'time res':
      - 'month'
      - 'day'
  'encoder':
      - 'one_hot'
      - 'ordinal'
  'classifier': 'logistic'
  'C':
      - 0.1
      - 1.0

```

</details>


# Keeping the original columns and renaming output columns

Sometimes we want to transform a column but still keep the original one in the output, maybe to transform it in a different way.
We can do it with `keep_original`:

```python
pipe.use(
    OneHotEncoder(sparse_output=False), cols="B", keep_original=False
)  # the default
```
<!-- output -->
```
<Pipe: 1 transformations>
Steps:
0: one_hot_encoder
Sample of transformed data:
   A           C    D    E  B_one  B_two
0  3  11/02/2012  2.5  7.2    0.0    1.0
1  1  01/02/1998  0.5  5.2    1.0    0.0
2  2  10/03/2027  1.5  6.2    1.0    0.0
3  4  23/04/1999  3.5  8.2    0.0    1.0
4  5  01/01/1901  4.5  9.2    0.0    1.0
```

```python
pipe.use(OneHotEncoder(sparse_output=False), cols="B", keep_original=True)
```
<!-- output -->
```
<Pipe: 1 transformations>
Steps:
0: one_hot_encoder
Sample of transformed data:
   A    B           C    D    E  B_one  B_two
0  3  two  11/02/2012  2.5  7.2    0.0    1.0
1  1  one  01/02/1998  0.5  5.2    1.0    0.0
2  2  one  10/03/2027  1.5  6.2    1.0    0.0
3  4  two  23/04/1999  3.5  8.2    0.0    1.0
4  5  two  01/01/1901  4.5  9.2    0.0    1.0
```

We can also rename the output columns.
For example this can be a way to insert a tag by which we can select them later.

```python
pipe.chain(
    s.cols("B").one_hot_encoder(sparse_output=False).rename_columns("<ohe-B>{}"),
    s.cols("A").one_hot_encoder(sparse_output=False).rename_columns("<ohe-A>{}"),
    s.glob("<ohe-[BA]>*").polynomial_features(
        degree=2, interaction_only=True, include_bias=False
    ),
).sample().iloc[0]
```
<!-- output -->
```
C                            11/02/2012
D                                   2.5
E                                   7.2
<ohe-B>B_one                        0.0
<ohe-B>B_two                        1.0
<ohe-A>A_1.0                        0.0
<ohe-A>A_2.0                        0.0
<ohe-A>A_3.0                        1.0
<ohe-A>A_4.0                        0.0
<ohe-A>A_5.0                        0.0
<ohe-B>B_one <ohe-B>B_two           0.0
<ohe-B>B_one <ohe-A>A_1.0           0.0
<ohe-B>B_one <ohe-A>A_2.0           0.0
<ohe-B>B_one <ohe-A>A_3.0           0.0
<ohe-B>B_one <ohe-A>A_4.0           0.0
<ohe-B>B_one <ohe-A>A_5.0           0.0
<ohe-B>B_two <ohe-A>A_1.0           0.0
<ohe-B>B_two <ohe-A>A_2.0           0.0
<ohe-B>B_two <ohe-A>A_3.0           1.0
<ohe-B>B_two <ohe-A>A_4.0           0.0
<ohe-B>B_two <ohe-A>A_5.0           0.0
<ohe-A>A_1.0 <ohe-A>A_2.0           0.0
<ohe-A>A_1.0 <ohe-A>A_3.0           0.0
<ohe-A>A_1.0 <ohe-A>A_4.0           0.0
<ohe-A>A_1.0 <ohe-A>A_5.0           0.0
<ohe-A>A_2.0 <ohe-A>A_3.0           0.0
<ohe-A>A_2.0 <ohe-A>A_4.0           0.0
<ohe-A>A_2.0 <ohe-A>A_5.0           0.0
<ohe-A>A_3.0 <ohe-A>A_4.0           0.0
<ohe-A>A_3.0 <ohe-A>A_5.0           0.0
<ohe-A>A_4.0 <ohe-A>A_5.0           0.0
Name: 0, dtype: object
```

# Hyperparam tuning example

```python
import pandas as pd
from sklearn import datasets
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 10)

from skrub._pipe import Pipe, choose, optional, choose_float

X, y = datasets.make_regression(random_state=0)

pipe = Pipe().chain(
    SelectKBest(k=choose(10, 20, 100).name("k"), score_func=f_regression),
    optional(StandardScaler()).name("rescale"),
    choose(
        ridge=Ridge(alpha=choose_float(0.01, 100.0, log=True).name("ridge.α")),
        lasso=Lasso(alpha=choose_float(0.1, 100.0, log=True).name("lasso.α")),
    ).name("regressor"),
)

X = pd.DataFrame(X, columns=map(str, range(X.shape[1])))
search = pipe.get_randomized_search(n_iter=32).fit(X, y)
print(pipe.get_cv_results_table(search))
```
<!-- output -->
```
    mean_score    k rescale regressor    ridge.α    lasso.α  fit_time  std_score
0     0.999622  100    true     lasso        NaN   0.635497  0.008475   0.000097
1     0.998522  100    true     lasso        NaN   1.256876  0.008649   0.000378
2     0.992439  100   false     lasso        NaN   2.671610  0.004763   0.001760
3     0.818178  100    true     lasso        NaN  15.262997  0.008437   0.060098
4     0.797651   20   false     lasso        NaN   0.374613  0.004518   0.114999
5     0.797088   20   false     lasso        NaN   1.903548  0.004518   0.107767
6     0.797002   20   false     lasso        NaN   0.139051  0.009248   0.117007
7     0.796481   20    true     ridge   0.030784        NaN  0.006571   0.118244
8     0.796429   20   false     ridge   0.043162        NaN  0.004492   0.118224
9     0.793053   20    true     ridge   0.924077        NaN  0.006470   0.117557
10    0.784482   20   false     ridge   2.521967        NaN  0.004412   0.115872
11    0.755150   20   false     lasso        NaN   8.175487  0.004438   0.093798
12    0.699595   10   false     lasso        NaN   0.197072  0.004444   0.200251
13    0.699532   10    true     lasso        NaN   0.938546  0.006331   0.198820
14    0.699367   10   false     ridge   0.022078        NaN  0.004609   0.200709
15    0.699357   10    true     ridge   0.028232        NaN  0.006200   0.200711
16    0.699350   10    true     ridge   0.031072        NaN  0.006181   0.200713
17    0.699345   10   false     ridge   0.031025        NaN  0.004434   0.200714
18    0.698547   10    true     lasso        NaN   2.613233  0.006371   0.195116
19    0.696782   10   false     ridge   0.995502        NaN  0.004356   0.201152
20    0.694957   10    true     ridge   1.750122        NaN  0.006298   0.201327
21    0.691919  100    true     ridge   0.488130        NaN  0.008358   0.128213
22    0.667502   10    true     ridge   9.533996        NaN  0.006345   0.198730
23    0.643554   10    true     ridge  15.476385        NaN  0.006209   0.193876
24    0.574500   10    true     lasso        NaN  20.463201  0.006267   0.164964
25    0.572064  100    true     ridge  13.862272        NaN  0.008459   0.113447
26    0.553455  100    true     ridge  17.295971        NaN  0.008444   0.109960
27    0.490367  100   false     ridge  28.827784        NaN  0.004769   0.097673
28    0.414345  100   false     lasso        NaN  29.954973  0.004693   0.136744
29    0.334967  100   false     lasso        NaN  34.409654  0.004734   0.122472
30    0.334361   10   false     lasso        NaN  34.455669  0.004579   0.122190
31    0.174537   10   false     lasso        NaN  47.765130  0.004447   0.065144
```
