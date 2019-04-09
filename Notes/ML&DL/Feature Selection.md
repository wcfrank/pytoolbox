# Feature Selection

特征选择是特征工程里的一个重要问题，目的是找到最优特征子集。减少特征个数，减少运行时间，提高模型精确度。

- Filter（过滤法）：按照发散性或者相关性对各个特征进行评分，设定阈值或者待选择阈值的个数，选择特征。
- Wrapper（包装法）：根据目标函数（通常是预测效果评分），每次选择若干特征，或者排除若干特征。
- Embedded（嵌入法）：先使用某些机器学习的算法和模型进行训练，得到各个特征的权值系数，根据系数从大到小排序选择特征。类似于Filter方法，但是是通过训练来确定特征的优劣。

## Filter特征选择

针对样本的每个特征$$x_i~(i=1,\dots,n)$$，计算$$x_i$$与label标签$$y$$的信息量$$S(i)$$，得到$$n$$个结果，选择前$$k$$个信息量最大的特征。即选取与$$y$$关联最密切的一些特征$$x_i$$。下面介绍几种度量$$S(i)$$的方法：

1. Pearson相关系数

   衡量变量之间的线性相关性，结果取值为$$[-1,1]$$，-1表示完全负相关，+1表示完全正相关，0表示没有**线性**相关。

   简单，计算速度快；但只对线性关系敏感，如果关系是非线性的，即使两个变量有关联，Pearson相关性也可能接近0。scipy的pearsonr方法能计算相关系数和p-value[2]：

   ```python
   import numpy as np
   from scipy.stats import pearsonr
   
   np.random.seed(0)
   size = 300
   x = np.random.normal(0, 1, size)
   print("Lower noise", pearsonr(x, x + np.random.normal(0, 1, size)))
   print("Higher noise", pearsonr(x, x + np.random.normal(0, 10, size)))
   ```

   output:

   (0.718248, 7.324017e-49), (0.057964, 0.317009)

2. 卡方验证（**常用**）

   检验自变量与因变量的相关性。假设自变量有N种取值，因变量有M种取值，自变量等于i且因变量等于j的样本频数的观察值与期望的差距：$$\chi^2 = \sum\frac{(A-E)^2}{E}$$.

   ```python
   from sklearn.datasets import load_iris
   from sklearn.feature_selection import SelectKBest
   from sklearn.feature_selection import chi2
   iris = load_iris()
   X, y = iris.data, iris.target
   #选择K个最好的特征，返回选择特征后的数据
   X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
   ```

   Output:  X.shape = (150,4), X_new.shape = (150,2)

3. **互信息和最大信息系数** Mutual information and maximal information coefficient (MIC)

## 参考资料

sklearn.feature_selection模块适用于样本的特征选择/维数降低

1. [特征选择](https://zhuanlan.zhihu.com/p/32749489)

2. [Statistical meaning of pearsonr() output in Python](https://stats.stackexchange.com/questions/64676/statistical-meaning-of-pearsonr-output-in-python)