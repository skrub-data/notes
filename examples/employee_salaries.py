# +
from skrub.datasets import fetch_employee_salaries

dataset = fetch_employee_salaries()
employees, salaries = dataset.X, dataset.y

# +
from skrub._pipe import Pipe
from skrub import TableVectorizer, MinHashEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

pipe = Pipe(employees)
pipe
# -

pipe.get_skrubview_report()

pipe = pipe.use(TableVectorizer(high_cardinality_transformer=MinHashEncoder()))
pipe = pipe.use(HistGradientBoostingRegressor())
pipe

pipe.get_skrubview_report()

# +
from sklearn.model_selection import cross_val_score

cross_val_score(pipe.pipeline, employees, salaries)
# -

# # gridsearch

# +
from skrub import DropCols
from skrub._pipe import choose
from sklearn.preprocessing import OneHotEncoder

pipe = (
    Pipe(employees)
    .use(DropCols(cols=choose("department", []).name("dropped cols")))
    .use(
        TableVectorizer(
            high_cardinality_transformer=MinHashEncoder(
                n_components=choose(5, 10).name("minhash dim")
            ),
            low_cardinality_transformer=choose(
                OneHotEncoder(sparse_output=False), "passthrough"
            ).name("low card encoder"),
        )
    )
    .use(HistGradientBoostingRegressor())
)
print(pipe.param_grid_description)
# -

gs = pipe.grid_search.fit(employees, salaries)
gs.best_params_

# +
print(pipe.get_best_params_description(gs))
