---
breaks: false
---

# A high-level interface for building a dataframe transformation pipeline

The goal is to have a more high-level and interactive interface for building a scikit-learn pipeline.
In particular it should offer:

- previews of a sample of the data transformed with the current pipeline.
- a way to very easily add steps that rely on one of the most commonly-used estimators such as `Ridge`, `HistGradientBoostingRegressor`.
- a way to add steps that rely on any scikit-learn compatible estimator
- a way to specify ranges of hyperparameters (for tuning) as the pipeline is being constructed.
- a way to perform cross-validation and hyperparameter search, eg by obtaining a scikit-learn GridSearchCV or  Pipeline and using scikit-learn cross-validation tools.

here is some toy data:

```
>>> import pandas as pd
>>> df = pd.DataFrame(
...     {
...         "A": list(range(1, 6)),
...         "B": "one one two two two".split(),
...         "C": ["01/02/1998", "10/03/2027", "11/02/2012", "23/04/1999", "01/01/1901"],
...         "D": [n + 0.5 for n in range(5)],
...         "E": [n + 5.2 for n in range(5)],
...     }
... ).convert_dtypes()

>>> df
   A    B           C    D    E
0  1  one  01/02/1998  0.5  5.2
1  2  one  10/03/2027  1.5  6.2
2  3  two  11/02/2012  2.5  7.2
3  4  two  23/04/1999  3.5  8.2
4  5  two  01/01/1901  4.5  9.2
>>> df.dtypes
A             Int64
B    string[python]
C    string[python]
D           Float64
E           Float64
dtype: object

```

# Applying some transformations

The pipeline is instantiated with a dataset so we can get the previews.

__Note:__ don't pay attention to the skrub imports for now; anything we decide to put in the public API will be importable directly from `skrub`.

```
>>> from skrub._pipe import Pipe, choose
>>> from skrub import selectors as s

>>> pipe = Pipe(df)
>>> pipe
<Pipe: 0 transformations>
Sample of transformed data:
   A    B           C    D    E
2  3  two  11/02/2012  2.5  7.2
0  1  one  01/02/1998  0.5  5.2
1  2  one  10/03/2027  1.5  6.2
3  4  two  23/04/1999  3.5  8.2
4  5  two  01/01/1901  4.5  9.2

```

Roughly 3 APIs for adding steps have been proposed; other suggestions welcome.

## Option 1: with `Pipe.use()`

In this option, the pipeline has a `use` method (can also be named `apply`, for example, if that's not too confusing with pandas' apply).
We pass it the transformer to use, and optional configuration such as the columns on which to use it or a name for the step as kwargs.

```
>>> from skrub._to_datetime import ToDatetime
>>> from skrub._datetime_encoder import EncodeDatetime
>>> from sklearn.preprocessing import OneHotEncoder
>>> from sklearn.linear_model import Ridge
>>> from skrub._to_numeric import ToNumeric

>>> (
...     p := pipe
...     .use(ToDatetime(), cols="C")
...     .use(EncodeDatetime(), cols=s.any_date(), name="encode-dt")
...     .use(OneHotEncoder(sparse_output=False), cols=s.string())
...     .use(Ridge())
... )
<Pipe: 3 transformations + Ridge>
Steps:
0: to_datetime, 1: encode-dt, 2: one_hot_encoder, 3: ridge
Sample of transformed data:
   A  C_year  C_month  C_day  C_total_seconds    D    E  B_one  B_two
0  3  2012.0      2.0   11.0     1.328918e+09  2.5  7.2    0.0    1.0
1  1  1998.0      2.0    1.0     8.862912e+08  0.5  5.2    1.0    0.0
2  2  2027.0      3.0   10.0     1.804637e+09  1.5  6.2    1.0    0.0
3  4  1999.0      4.0   23.0     9.248256e+08  3.5  8.2    0.0    1.0
4  5  1901.0      1.0    1.0    -2.177453e+09  4.5  9.2    0.0    1.0

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

We can then extract a scikit-learn Pipeline that we can cross-validate etc.

```
>>> p.pipeline
Pipeline(steps=[('to_datetime',
                 OnEachColumn(cols=['C'], transformer=ToDatetime())),
                ('encode-dt',
                 OnEachColumn(cols=any_date(), transformer=EncodeDatetime())),
                ('one_hot_encoder',
                 OnColumnSelection(cols=string(), transformer=OneHotEncoder(sparse_output=False))),
                ('ridge', Ridge())])

```

## Option 2: with `Selector.to_datetime()`, `Selector.use()`

We can also have the `use` method directly on the selectors, some methods for commonly used estimators.
This last point avoids having to import estimators and provides tab-completion on their name in interactive shells.
Another difference is configuration like the step name is added with an additional method call rather than additional kwargs.

```
>>> pipe.chain(
...     s.cols("C").to_datetime(),
...     s.any_date().encode_datetime().name("encode-dt"),
...     s.string().one_hot_encoder(sparse_output=False),
...     s.all().ridge(),
... )
<Pipe: 3 transformations + Ridge>
Steps:
0: to_datetime, 1: encode-dt, 2: one_hot_encoder, 3: ridge
Sample of transformed data:
   A  C_year  C_month  C_day  C_total_seconds    D    E  B_one  B_two
0  3  2012.0      2.0   11.0     1.328918e+09  2.5  7.2    0.0    1.0
1  1  1998.0      2.0    1.0     8.862912e+08  0.5  5.2    1.0    0.0
2  2  2027.0      3.0   10.0     1.804637e+09  1.5  6.2    1.0    0.0
3  4  1999.0      4.0   23.0     9.248256e+08  3.5  8.2    0.0    1.0
4  5  1901.0      1.0    1.0    -2.177453e+09  4.5  9.2    0.0    1.0

```
(Instead of `chain` it could be `apply`, `use`, `transform`, `with_steps`, ...)

We can also pass directly an estimator (in which case the column selection is `s.all()`), and the selectors have a `.use()` method for using estimators that haven't been registered as methods.
So this is equivalent to the above:

```
>>> pipe.chain(
...     s.cols("C").to_datetime(),
...     s.any_date().encode_datetime().name("encode-dt"),
...     s.string().use(OneHotEncoder(sparse_output=False)),
...     Ridge(),
... )
<Pipe: 3 transformations + Ridge>
Steps:
0: to_datetime, 1: encode-dt, 2: one_hot_encoder, 3: ridge
Sample of transformed data:
   A  C_year  C_month  C_day  C_total_seconds    D    E  B_one  B_two
0  3  2012.0      2.0   11.0     1.328918e+09  2.5  7.2    0.0    1.0
1  1  1998.0      2.0    1.0     8.862912e+08  0.5  5.2    1.0    0.0
2  2  2027.0      3.0   10.0     1.804637e+09  1.5  6.2    1.0    0.0
3  4  1999.0      4.0   23.0     9.248256e+08  3.5  8.2    0.0    1.0
4  5  1901.0      1.0    1.0    -2.177453e+09  4.5  9.2    0.0    1.0

```

## Option3: with `Pipe.cols().to_datetime()`, `Pipe.cols().use()`

The third option adds a method `.cols` (or maybe `.on_cols`) to the pipeline to which we pass the selector.
That returns an object that is used to configure the next step

```
>>> (
...     pipe
...     .cols("C").to_datetime()
...     .cols(s.any_date()).encode_datetime()
...     .cols(s.string()).one_hot_encoder(sparse_output=False)
...     .cols(s.all()).ridge()
... )
<Pipe: 3 transformations + Ridge>
Steps:
0: to_datetime, 1: encode_datetime, 2: one_hot_encoder, 3: ridge
Sample of transformed data:
   A  C_year  C_month  C_day  C_total_seconds    D    E  B_one  B_two
0  3  2012.0      2.0   11.0     1.328918e+09  2.5  7.2    0.0    1.0
1  1  1998.0      2.0    1.0     8.862912e+08  0.5  5.2    1.0    0.0
2  2  2027.0      3.0   10.0     1.804637e+09  1.5  6.2    1.0    0.0
3  4  1999.0      4.0   23.0     9.248256e+08  3.5  8.2    0.0    1.0
4  5  1901.0      1.0    1.0    -2.177453e+09  4.5  9.2    0.0    1.0


```

Notes:

Methods that add an estimator (eg `encode_datetime()`) have to return the `Pipe` object itself, so it's not clear where we should provide configuration such as the step name.
That may not be very important, as ATM I don't see anything else than the step name to configure (there could be a `param_grid` but the other way of specifying it described later seems better), and the step name may not be that important.

We could also say that there is a `.name()` method on the `Pipe` itself that implicitly applies to the last step.

We cannot pass an estimator directly, but the result of `cols` has a `use()` (or `apply()`, or ...) method:

```
>>> (
...     pipe
...     .cols("C").use(ToDatetime())
...     .cols(s.any_date()).use(EncodeDatetime())
...     .cols(s.string()).use(OneHotEncoder(sparse_output=False))
...     .cols(s.all()).use(Ridge())
... )
<Pipe: 3 transformations + Ridge>
Steps:
0: to_datetime, 1: encode_datetime, 2: one_hot_encoder, 3: ridge
Sample of transformed data:
   A  C_year  C_month  C_day  C_total_seconds    D    E  B_one  B_two
0  3  2012.0      2.0   11.0     1.328918e+09  2.5  7.2    0.0    1.0
1  1  1998.0      2.0    1.0     8.862912e+08  0.5  5.2    1.0    0.0
2  2  2027.0      3.0   10.0     1.804637e+09  1.5  6.2    1.0    0.0
3  4  1999.0      4.0   23.0     9.248256e+08  3.5  8.2    0.0    1.0
4  5  1901.0      1.0    1.0    -2.177453e+09  4.5  9.2    0.0    1.0

```

As the `.cols()` looks like we are indexing the data it may be a bit surprising if someone expects the result of the transformation on just those columns to be returned:

```
>>> p = pipe.cols(["A", "B"]).one_hot_encoder(sparse_output=False)
>>> p
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

## Discarded options

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

```
>>> from skrub._pipe import choose

>>> (
...     p := pipe
...     .use(ToDatetime(), cols="C")
...     .use(EncodeDatetime(resolution=choose("month", "day").name("time res")),
...          cols=s.any_date())
...     .use(OneHotEncoder(sparse_output=False), cols=s.string())
...     .use(Ridge(alpha=choose(1.0, 10.0, 100.0).name("α")))
... )
<Pipe: 3 transformations + Ridge>
Steps:
0: to_datetime, 1: encode_datetime, 2: one_hot_encoder, 3: ridge
Sample of transformed data:
   A  C_year  C_month  C_total_seconds    D    E  B_one  B_two
0  3  2012.0      2.0     1.328918e+09  2.5  7.2    0.0    1.0
1  1  1998.0      2.0     8.862912e+08  0.5  5.2    1.0    0.0
2  2  2027.0      3.0     1.804637e+09  1.5  6.2    1.0    0.0
3  4  1999.0      4.0     9.248256e+08  3.5  8.2    0.0    1.0
4  5  1901.0      1.0    -2.177453e+09  4.5  9.2    0.0    1.0

>>> print(p.param_grid_description)
- 'encode_datetime': <EncodeDatetime(resolution=<time res>).transform(col) for col in X[any_date()]>
  'time res':
      - month
      - day
  'ridge': Ridge(alpha=<α>)
  'α':
      - 1.0
      - 10.0
      - 100.0

>>> p.grid_search
GridSearchCV(estimator=Pipeline(steps=[('to_datetime',
                                        OnEachColumn(cols=['C'], transformer=ToDatetime())),
                                       ('encode_datetime',
                                        OnEachColumn(cols=any_date(), transformer=EncodeDatetime(resolution='month'))),
                                       ('one_hot_encoder',
                                        OnColumnSelection(cols=string(), transformer=OneHotEncoder(sparse_output=False))),
                                       ('ridge', Ridge())]),
             param_grid=[{'encode_datetime': choose(<EncodeDatetime(resolution=<time res>).transform(col) for col in X[any_date()]>),
                          'encode_datetime__transformer__resolution': choose('month', 'day').name('time res'),
                          'ridge': choose(Ridge(alpha=<α>)),
                          'ridge__alpha': choose(1.0, 10.0, 100.0).name('α')}])

```

<details>
<summary>hyperparameter choice with the alternative APIs</summary>

## `Selector.use`

```
>>> p = pipe.chain(
...     s.cols("C").to_datetime(),
...     s.any_date().encode_datetime(resolution=choose("month", "day").name("time res")),
...     s.string().one_hot_encoder(sparse_output=False),
...     s.all().ridge(alpha=choose(1.0, 10.0).name("α")),
... )

>>> print(p.param_grid_description)
- 'encode_datetime': <EncodeDatetime(resolution=<time res>).transform(col) for col in X[any_date()]>
  'time res':
      - month
      - day
  'ridge': Ridge(alpha=<α>)
  'α':
      - 1.0
      - 10.0

```

## `Pipe.cols`

```
>>> p = (
...     pipe
...     .cols("C").to_datetime()
...     .cols(s.any_date()).encode_datetime(resolution=choose("month", "day").name("time res"))
...     .cols(s.string()).one_hot_encoder(sparse_output=False)
...     .cols(s.all()).ridge(alpha=choose(1.0, 10.0).name("α"))
... )

>>> print(p.param_grid_description)
- 'encode_datetime': <EncodeDatetime(resolution=<time res>).transform(col) for col in X[any_date()]>
  'time res':
      - month
      - day
  'ridge': Ridge(alpha=<α>)
  'α':
      - 1.0
      - 10.0


```
</details>

# Choosing among several estimators

We may also want to choose among several estimators.
We can pass a `Choice` to `use`: `pipe.use(choose(Ridge(), LogisticRegression()))`.
For convenience, to remove one nested call, the `Pipe` can also have a `choose` method:
`pipe.choose(opt1, opt2)` is shorthand for `pipe.use(choose(opt1, opt2))`.

```
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.preprocessing import OrdinalEncoder

>>> (
...     p := pipe.use(ToDatetime(), cols="C")
...     .use(EncodeDatetime(resolution=choose("month", "day").name("time res")),
...          cols=s.any_date())
...     .choose(
...         OneHotEncoder(sparse_output=False),
...         OrdinalEncoder(),
...         cols=s.string(),
...         name="cat-encoder",
...      )
...     .choose(
...         Ridge(alpha=choose(1.0, 10.0).name("α")),
...         LogisticRegression(C=choose(0.1, 1.0).name("C")),
...         name="regressor",
...      )
... )
<Pipe: 3 transformations + Ridge>
Steps:
0: to_datetime, 1: encode_datetime, 2: cat-encoder, 3: regressor
Sample of transformed data:
   A  C_year  C_month  C_total_seconds    D    E  B_one  B_two
0  3  2012.0      2.0     1.328918e+09  2.5  7.2    0.0    1.0
1  1  1998.0      2.0     8.862912e+08  0.5  5.2    1.0    0.0
2  2  2027.0      3.0     1.804637e+09  1.5  6.2    1.0    0.0
3  4  1999.0      4.0     9.248256e+08  3.5  8.2    0.0    1.0
4  5  1901.0      1.0    -2.177453e+09  4.5  9.2    0.0    1.0

>>> print(p.param_grid_description)
- 'encode_datetime': <EncodeDatetime(resolution=<time res>).transform(col) for col in X[any_date()]>
  'time res':
      - month
      - day
  'cat-encoder':
      - <OneHotEncoder(sparse_output=False).transform(X[string()])>
      - <OrdinalEncoder().transform(X[string()])>
  'regressor': Ridge(alpha=<α>)
  'α':
      - 1.0
      - 10.0
- 'encode_datetime': <EncodeDatetime(resolution=<time res>).transform(col) for col in X[any_date()]>
  'time res':
      - month
      - day
  'cat-encoder':
      - <OneHotEncoder(sparse_output=False).transform(X[string()])>
      - <OrdinalEncoder().transform(X[string()])>
  'regressor': LogisticRegression(C=<C>)
  'C':
      - 0.1
      - 1.0

```

<details>
<summary>With the allternative APIs</summary>

## `Selector.use`

```
>>> p = pipe.chain(
...     s.cols("C").to_datetime(),
...     s.any_date().encode_datetime(resolution=choose("month", "day").name("time res")),
...     s.string().choose(
...         OneHotEncoder(sparse_output=False),
...         OrdinalEncoder()
...     ).name("encoder"),
...     choose(
...         Ridge(alpha=choose(1.0, 10.0).name("α")),
...         LogisticRegression(C=choose(0.1, 1.0).name("C")),
...     ).name("regressor"),
... )

>>> print(p.param_grid_description)
- 'encode_datetime': <EncodeDatetime(resolution=<time res>).transform(col) for col in X[any_date()]>
  'time res':
      - month
      - day
  'encoder':
      - <OneHotEncoder(sparse_output=False).transform(X[string()])>
      - <OrdinalEncoder().transform(X[string()])>
  'regressor': Ridge(alpha=<α>)
  'α':
      - 1.0
      - 10.0
- 'encode_datetime': <EncodeDatetime(resolution=<time res>).transform(col) for col in X[any_date()]>
  'time res':
      - month
      - day
  'encoder':
      - <OneHotEncoder(sparse_output=False).transform(X[string()])>
      - <OrdinalEncoder().transform(X[string()])>
  'regressor': LogisticRegression(C=<C>)
  'C':
      - 0.1
      - 1.0

```

## `Pipe.cols`

```
>>> p = (
...     pipe
...     .cols("C").to_datetime()
...     .cols(s.any_date()).encode_datetime(resolution=choose("month", "day").name("time res"))
...     .cols(s.string()).choose(
...         OneHotEncoder(sparse_output=False),
...         OrdinalEncoder()
...     )
...     .cols(s.all()).choose(
...         Ridge(alpha=choose(1.0, 10.0).name("α")),
...         LogisticRegression(C=choose(0.1, 1.0).name("C")),
...     )
... )
>>> print(p.param_grid_description)
- 'encode_datetime': <EncodeDatetime(resolution=<time res>).transform(col) for col in X[any_date()]>
  'time res':
      - month
      - day
  'one_hot_encoder':
      - <OneHotEncoder(sparse_output=False).transform(X[string()])>
      - <OrdinalEncoder().transform(X[string()])>
  'ridge': Ridge(alpha=<α>)
  'α':
      - 1.0
      - 10.0
- 'encode_datetime': <EncodeDatetime(resolution=<time res>).transform(col) for col in X[any_date()]>
  'time res':
      - month
      - day
  'one_hot_encoder':
      - <OneHotEncoder(sparse_output=False).transform(X[string()])>
      - <OrdinalEncoder().transform(X[string()])>
  'ridge': LogisticRegression(C=<C>)
  'C':
      - 0.1
      - 1.0

```
</details>
