import pandas as pd
import skrub
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate


#########################################
# Download the data
#########################################
from relbench.datasets import get_dataset
dataset = get_dataset(name="rel-stackex")
task = dataset.get_task("rel-stackex-votes")


#########################################
# Simple GDT baseline
#########################################

whole_df = pd.concat([task.train_table.df,
                      task.val_table.df,
                      task.test_table.df], axis=0)

# Drop the rows were y is missing
whole_df = whole_df.dropna(subset=task.target_col)

y = whole_df[task.target_col]
X = whole_df.drop(task.target_col, axis=1)

gbt = make_pipeline(skrub.TableVectorizer(),
                    HistGradientBoostingRegressor())

#gbt_scores = cross_validate(gbt, X, y, n_jobs=-1, verbose=10)
#print("Simple baseline; R2 score %.3f " % gbt_scores['test_score'].mean())

#########################################
# Merge multiple tables for learning
#########################################

# All the tables can be found in dataset.db.table_dict

entity_df = dataset.db.table_dict[task.entity_table].df

merged_df = whole_df.merge(
        entity_df,
        how="left",
        left_on='PostId',
        right_on='Id',
    )

y = merged_df[task.target_col]
X = merged_df.drop(task.target_col, axis=1)

gbt = make_pipeline(
        skrub.TableVectorizer(
                verbose=3,
                high_cardinality_transformer=skrub.MinHashEncoder(n_components=30)),
        HistGradientBoostingRegressor(verbose=3))

#gbt_scores = cross_validate(gbt, X, y, n_jobs=1, verbose=10)
#print("Merged baseline; R2 score %.3f " % gbt_scores['test_score'].mean())


#########################################
# Use featuretools
#########################################
import featuretools as ft
tables = ft.EntitySet(id='stackex')


# main table is one of popularity at different time stamps on posts.
# make_index is needed to add a column that is unique
tables.add_dataframe(dataframe=whole_df,
                     dataframe_name='target',
                     make_index=True,
                     index='index',
                    )


for this_name, this_df in dataset.db.table_dict.items():
    this_df = this_df.df
    # featuretools cannot deal with NAs in keys. We replace by '-1'
    for col in this_df.columns:
        if col.endswith('Id'):
            this_df[col] = this_df[col].fillna(value=-1)
    tables.add_dataframe(dataframe=this_df,
                        dataframe_name=this_name,
                    )


# Add the links across tables

# Column in parent must be an index
tables.add_relationship(
    parent_dataframe_name='posts',
    parent_column_name='Id',
    child_dataframe_name='postLinks',
    child_column_name='PostId',
)

tables.add_relationship(
    parent_dataframe_name='posts',
    parent_column_name='Id',
    child_dataframe_name='postLinks',
    child_column_name='RelatedPostId',
)

tables.add_relationship(
    parent_dataframe_name='posts',
    parent_column_name='Id',
    child_dataframe_name='target',
    child_column_name='PostId',
)

tables.add_relationship(
    parent_dataframe_name='posts',
    parent_column_name='Id',
    child_dataframe_name='postHistory',
    child_column_name='PostId',
)

tables.add_relationship(
    parent_dataframe_name='posts',
    parent_column_name='Id',
    child_dataframe_name='comments',
    child_column_name='PostId',
)

tables.add_relationship(
    parent_dataframe_name='posts',
    parent_column_name='Id',
    child_dataframe_name='votes',
    child_column_name='PostId',
)

tables.add_relationship(
    parent_dataframe_name='posts',
    parent_column_name='Id',
    child_dataframe_name='posts',
    child_column_name='ParentId',
)

tables.add_relationship(
    parent_dataframe_name='posts',
    parent_column_name='Id',
    child_dataframe_name='posts',
    child_column_name='AcceptedAnswerId',
)

tables.add_relationship(
    parent_dataframe_name='users',
    parent_column_name='Id',
    child_dataframe_name='postHistory',
    child_column_name='UserId',
)


tables.add_relationship(
    parent_dataframe_name='users',
    parent_column_name='Id',
    child_dataframe_name='comments',
    child_column_name='UserId',
)


tables.add_relationship(
    parent_dataframe_name='users',
    parent_column_name='Id',
    child_dataframe_name='badges',
    child_column_name='UserId',
)


tables.add_relationship(
    parent_dataframe_name='users',
    parent_column_name='Id',
    child_dataframe_name='votes',
    child_column_name='UserId',
)

tables.add_relationship(
    parent_dataframe_name='users',
    parent_column_name='Id',
    child_dataframe_name='posts',
    child_column_name='OwnerUserId',
)

# If you have graphviz install, visualize as such
#dig = tables.plot()
#dig.render()

