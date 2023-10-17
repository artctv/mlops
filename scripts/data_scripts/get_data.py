from catboost.datasets import titanic

train, test = titanic()

train.to_csv("/root/mlops/data/raw/train.csv", columns=train.columns)
test.to_csv("/root/mlops/data/raw/test.csv", columns=test.columns)
