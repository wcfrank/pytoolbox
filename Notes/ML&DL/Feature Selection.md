# Feature Selection

特征选择是特征工程里的一个重要问题，目的是找到最优特征子集。**减少特征个数**，减少运行时间，提高模型精确度；**更好的理解特征**，及其与label之间的相关性。

- Filter（过滤法）：按照发散性或者相关性对各个特征进行评分，设定阈值或者待选择阈值的个数，选择特征。
- Wrapper（包装法）：根据目标函数（通常是预测效果评分），每次选择若干特征，或者排除若干特征。
- Embedded（嵌入法）：先使用某些机器学习的算法和模型进行训练，得到各个特征的权值系数，根据系数从大到小排序选择特征。类似于Filter方法，但是是通过训练来确定特征的优劣。

## Filter特征选择(Univariate selection)

针对样本的每个特征$$x_i~(i=1,\dots,n)$$，计算$$x_i$$与label标签$$y$$的信息量$$S(i)$$，得到$$n$$个结果，选择前$$k$$个信息量最大的特征。即选取与$$y$$关联最密切的一些特征$$x_i$$。下面介绍几种度量$$S(i)$$的方法：

1. Pearson相关系数

   衡量变量之间的线性相关性(linear correlation)，结果取值为$$[-1,1]$$，-1表示完全负相关，+1表示完全正相关，0表示没有**线性**相关。

   简单，计算速度快；但只对线性关系敏感，如果关系是非线性的，即使两个变量有关联，Pearson相关性也可能接近0。scipy的pearsonr方法能计算相关系数和p-value[2], roughly showing the probability of an uncorrelated system creating a correlation value of this magnitude. The p-value is high meaning that it is very likely to observe such correlation on a dataset of this size purely by chance[6]：

   ```python
   import numpy as np
   from scipy.stats import pearsonr
   
   np.random.seed(0)
   size = 300
   x = np.random.normal(0, 1, size)
   print("Lower noise", pearsonr(x, x + np.random.normal(0, 1, size)))
   print("Higher noise", pearsonr(x, x + np.random.normal(0, 10, size)))
   # output: (0.718248, 7.324017e-49), (0.057964, 0.317009)
   ```

   类似的，在sklearn中针对回归问题有`f_regression`函数，测量一组变量与label的线性关系的p-value[6]
   
   Relying only on the correlation value on interpreting the relationship of two variables can be highly misleading, so it is always worth plotting the data[6]
   

3. **互信息和最大信息系数** Mutual information and maximal information coefficient (MIC)

   MI评价自变量与因变量的相关性。当$$x_i$$为0/1取值时，$$MI(x_i,y) = \sum\limits_{x_i\in\{0,1\}}\sum\limits_{y\in\{0,1\}}p(x_i,y)\log\frac{p(x_i,y)}{p(x_i)p(y)}$$，同理也很容易推广到多个离散值情形。可以发现MI衡量的是$$x_i$$和$$y$$的独立性，如果两者独立，MI=0，即$$x_i$$和$$y$$不相关，可以去除$$x_i$$；反之两者相关，MI会很大。

   MI的缺点：不属于度量方式，无法归一化；无法计算连续值特征，通常需要先离散化，但对离散化方式很敏感。

   MIC解决MI的缺点：首先，寻找最优的离散化方式；然后，把MI变成一种度量方式，区间为$$[0,1]$$
   ```python
   from minepy import MINE
   m = MINE()
   x = np.random.uniform(-1, 1, 10000)
   m.compute_score(x, x**2)
   print m.mic() # output: 1, the maximum
   ```
2. Distance Correlation

   Pearson相较MIC或者Distance correlation的优势：1. 计算速度快；2. correlaiton的取值区间是[-1,1]，体现正负相关性
   
4. 卡方验证（**常用**）

   检验自变量与因变量的相关性。假设自变量有N种取值，因变量有M种取值，自变量等于i且因变量等于j的样本频数的观察值与期望的差距：$$\chi^2 = \sum\frac{(A-E)^2}{E}$$.

   ```python
   from sklearn.datasets import load_iris
   from sklearn.feature_selection import SelectKBest
   from sklearn.feature_selection import chi2
   iris = load_iris()
   X, y = iris.data, iris.target
   #选择K个最好的特征，返回选择特征后的数据
   X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
   # Output:  X.shape = (150,4), X_new.shape = (150,2)
   ```



5. Variance Threshold

   但这种方法不需要度量特征$$x_i$$和标签$$y$$的关系。计算各个特征的方差，然后根据阈值选择方差大于阈值的特征。

   ```python
   from sklearn.feature_selection import VarianceThreshold
   X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
   sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
   print(sel.fit_transform(X))
   ```

   

## Wrapper特征选择[1]

在确定模型之后，不断的使用不同的特征组合来测试模型的表现，一般选用普遍效果较好的算法，如RF，SVM，kNN等。

- 前向搜索：每次从未选中的特征集合中选出一个加入，直到达到阈值或n为止

- 后向搜索：每一步删除一个特征

- 递归特征消除法RFE：使用一个模型进行多轮训练，每轮训练后消除某些特征，再基于新特征进行下一轮训练。

  ```python
  from sklearn.feature_selection import RFE
  from sklearn.linear_model import LogisticRegression
  #递归特征消除法，返回特征选择后的数据
  #参数estimator为基模型
  #参数n_features_to_select为选择的特征个数
  rfe = RFE(estimator=LogisticRegression(), n_features_to_select=2, verbose=3)
  rfe.fit(iris.data, iris.target)
  rfe.ranking_
  ```
  [4]：

  ```python
  from sklearn.feature_selection import RFECV
  all_features = [...]
  rfr = RandomForestRegressor(n_estimators=100, max_features='sqrt', max_depth=12, n_jobs=-1)
  rfecv = RFECV(estimator=rfr, step=10, 
                cv=KFold(y.shape[0], n_folds=5, shuffle=False, random_state=101),
                scoring='neg_mean_absolute_error', verbose=2)
  rfecv.fit(X, y)
  sel_features = [f for f, s in zip(all_features, rfecv.support_) if s]
  
  print(' Optimal number of features: %d' % rfecv.n_features_)
  print(' The selected features are {}'.format(sel_features))
  
  # Save sorted feature rankings
  ranking = pd.DataFrame({'Features': all_features})
  ranking['Rank'] = np.asarray(rfecv.ranking_)
  ranking.sort_values('Rank', inplace=True)
  ```

  

## Embedded特征选择

- 基于惩罚项：

  L1正则项具有稀疏解的特性，适合特征选择。但L1没选到的特征不代表不重要，因为两个高相关性的特征可能只保留了一个。如果要确定哪个特征重要，再通过L2正则交叉验证。

- Linear Model:

  sklearn可使用线性模型的.coef_来返回线性模型训练后的特征权重

  ```python
  lr = LinearRegression(normalize=True)
  lr.fit(X,Y)
  print(np.abs(lr.coef_))
  
  ridge = Ridge(alpha = 7)
  ridge.fit(X,Y)
  print(np.abs(ridge.coef_)
  
  lasso = Lasso(alpha=.05)
  lasso.fit(X, Y)
  print(lasso.coef_)
  ```

- Random Forest: .feature_importances_

- 基于模型的特征排序：

  直接用你要用的模型，对每个**单独**特征和标签$$y$$建立模型。假设此特征和标签的关系是非线形的，可用tree based模型，因为他们适合非线形关系的模型，但要注意防止过拟合，树的深度不要大，并运用交叉验证。

  ```python
  from sklearn.cross_validation import cross_val_score, ShuffleSplit
  from sklearn.datasets import load_boston
  from sklearn.ensemble import RandomForestRegressor
  
  boston = load_boston()
  X = boston["data"]
  Y = boston["target"]
  names = boston["feature_names"]
  
  rf = RandomForestRegressor(n_estimators=20, max_depth=4)
  scores = []
  for i in range(X.shape[1]):
       #每次选择一个特征，进行交叉验证，训练集和测试集为7:3的比例进行分配，
       #ShuffleSplit()函数用于随机抽样（数据集总数，迭代次数，test所占比例）
       score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",
                                cv=ShuffleSplit(len(X), 3, .3))
       scores.append((round(np.mean(score), 3), names[i]))
  
  #打印出各个特征所对应的得分
  print(sorted(scores, reverse=True))
  ```

  

## 参考资料

sklearn.feature_selection模块适用于样本的特征选择/维数降低

1. [特征选择](https://zhuanlan.zhihu.com/p/32749489)

2. [Statistical meaning of pearsonr() output in Python](https://stats.stackexchange.com/questions/64676/statistical-meaning-of-pearsonr-output-in-python)

3. (kaggle)[Feature Ranking RFE, Random Forest, linear models](https://www.kaggle.com/arthurtok/feature-ranking-rfe-random-forest-linear-models)

   Compare different kinds of feature ranking: Stability selection, recursive feature elimination, linear model, random forest feature ranking. Then create a feature ranking matrix, each column presents one feature ranking, using the function `ranking` to scale the ranking from 0 to 1. 

   最后，seaborn的pariplot(feature distribution)、heatmap(feature correlation)和factorplot(catplot)很漂亮。


4. [Recursive feature elimination](https://www.kaggle.com/tilii7/recursive-feature-elimination/code)
5. [Boruta feature elimination](https://www.kaggle.com/tilii7/boruta-feature-elimination)
6. **精华**[Feature selection – Part I: univariate selection](https://blog.datadive.net/selecting-good-features-part-i-univariate-selection/)
    
    Univariate selection examines each feature individually to determine the strength of the relationship of the feature with the lable
