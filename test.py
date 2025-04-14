from pathlib import Path
import pandas as pd
import numpy as np
import tarfile
import urllib.request
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

"""获取数据"""
def load_housing_data() -> pd.DataFrame:  
    return pd.read_csv("./datasets/housing/housing.csv")

housing = load_housing_data()

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(train_set.head(3))

"""计算分箱"""
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3., 4.5, 6, np.inf],
                               labels=[1, 2, 3, 4, 5])
housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)

"""分层拆分"""
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
strat_splits: list = []

for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])
    
strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, random_state=42, stratify=housing["income_cat"] 
)
strat_test_set["income_cat"].value_counts() / len(strat_test_set)

for set_ in (strat_test_set, strat_train_set):
    set_.drop("income_cat", axis=1, inplace=True)
    
print(strat_test_set)
