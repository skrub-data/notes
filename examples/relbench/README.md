# RelBench: learning benchmark on multi-table tables

## Exploring RelBench

https://relbench.stanford.edu/databases/stack_exchange/

2 datasets:
* stackex: 800Mb, 7 tables
* amazon: 8Gb, 3 tables

Stackex seems a good source of example data for skrub

### Downloading the data

The hashes computed by the code are wrong on my computer. 

Solution: edit the `__init__.py` of the relbench package and in the
`registry`, replace all the hashes by `None` as below:

```
    registry={
        # extremely small dataset only used for testing download functionality
        "rel-amazon-fashion_5_core/db.zip": None,
```

### Using with feature tools

Many merges are needed, this is tedious. We can use feature tools to
partly automate

Note: links must be specified to featuretools
