# Feature importance

## RF feature importance

All tree-based models have `feature_importances_`, like RandomForestClassifier (xgboost, lightgbm). For classification, it is typically either [Gini impurity](http://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity) or [information gain/entropy](http://en.wikipedia.org/wiki/Information_gain_in_decision_trees) and for regression trees it is [variance](http://en.wikipedia.org/wiki/Variance). Thus when training a tree, it can be computed how much each feature decreases the weighted impurity in a tree. For a forest, the impurity decrease from each feature can be averaged and the features are ranked according to this measure. [2]

There are a few things to keep in mind when using the impurity based ranking:  [2]

1. **feature selection based on impurity reduction is biased towards preferring variables with more categories** (see [Bias in random forest variable importance measures](http://link.springer.com/article/10.1186%2F1471-2105-8-25)). 

2.  when the dataset has two (or more) correlated features, then from the point of view of the model, any of these correlated features can be used as the predictor, with no concrete preference of one over the others. But once one of them is used, the importance of others is significantly reduced since effectively the impurity they can remove is already removed by the first feature. As a consequence, they will have a lower reported importance. 



## Boruta

[3]

shadow feature



## Permutation Importance (Mean decrease accuracy)

The general idea is to permute the values of each feature and measure how much the permutation decreases the accuracy of the model.

1. Get a trained model
2. Shuffle the values in a single column, make predictions using the resulting dataset. Use these predictions and the true target values to calculate how much the loss function suffered from shuffling. That performance deterioration measures the importance of the variable you just shuffled.
3. Return the data to the original order (undoing the shuffle from step 2.) Now repeat step 2 with the next column in the dataset, until you have calculated the importance of each column.

**Permutation importance is calculated after a model has been fitted.** So we won't change the model or change what predictions we'd get for a given value of height, sock-count, etc. [1]



#### Code example [1]

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('../input/fifa-2018-match-statistics/FIFA 2018 Statistics.csv')
y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary
feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)

import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())
```

The values towards the top are the most important features, and those towards the bottom matter least.



#### Code example [2]

```python
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict
X = boston["data"]
Y = boston["target"]
rf = RandomForestRegressor()
scores = defaultdict(list)

#crossvalidate the scores on a number of different random splits of the data
for train_idx, test_idx in ShuffleSplit(len(X), 100, .3):
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    r = rf.fit(X_train, Y_train)
    acc = r2_score(Y_test, rf.predict(X_test))
    for i in range(X.shape[1]):
        X_t = X_test.copy()
        np.random.shuffle(X_t[:, i])
        shuff_acc = r2_score(Y_test, rf.predict(X_t))
        scores[names[i]].append((acc-shuff_acc)/acc)
print "Features sorted by their score:"
print sorted([(round(np.mean(score), 4), feat) for feat, score in scores.items()], reverse=True)
```

Outputs: Features sorted by their score
[(0.7276, 'LSTAT'), (0.5675, 'RM'), (0.0867, 'DIS'), (0.0407, 'NOX'), (0.0351, 'CRIM'), (0.0233, 'PTRATIO'), (0.0168, 'TAX'), (0.0122, 'AGE'), (0.005, 'B'), (0.0048, 'INDUS'), (0.0043, 'RAD'), (0.0004, 'ZN'), (0.0001, 'CHAS')]

`LSTAT` and `RM` are two features that strongly impact model performance: permuting them decreases model performance by ~73% and ~57% respectively. Keep in mind that these measurements are made only after the model has been trained (and is depending) on all of these features. 

Reference:

1. https://www.kaggle.com/dansbecker/permutation-importance

2. [Selecting good features – Part III: random forests](https://blog.datadive.net/selecting-good-features-part-iii-random-forests/)

3. [Feature Importance Measures for Tree Models — Part I](https://medium.com/the-artificial-impostor/feature-importance-measures-for-tree-models-part-i-47f187c1a2c3)

   - https://www.kaggle.com/ogrellier/noise-analysis-of-porto-seguro-s-features

   - https://www.kaggle.com/ogrellier/feature-selection-target-permutations (use target permutation instead of feature permutation)

   - http://danielhomola.com/2015/05/08/borutapy-an-all-relevant-feature-selection-method/

     

