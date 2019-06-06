# 交叉熵

## 交叉熵定义

熵用来衡量随机变量的确定性，熵越大，变量的取值越不确定；反之，熵越小，变量取值就越确定。熵的公式：$H(x) = -\sum\limits_{x\in X}p(x)\log p(x)$，注意负号。

### 相对熵（KL散度）

两个随机分布间距离的度量，记为$D_{KL}(p||q)$。$D_{KL}(p||q) = \sum\limits_{x\in X}p(x)\log\frac{p(x)}{q(x)}$。当p=q时，KL散度最小，等于0. 

> 相对熵的意义就很明确了：$D_{KL}(p||q)$表示在真实分布为p的前提下，使用q分布进行编码相对于使用真实分布p进行编码（即最优编码）所多出来的bit数。[3]

### 交叉熵

> 交叉熵描述了两个概率分布之间的距离，交叉熵越小说明两者之间越接近。[2]

交叉熵容易跟相对熵搞混，二者有所区别。 假设有两个分布p，q，它们在给定样本集上的交叉熵定义如下：

$CEH(p,q)=−\sum_{x∈χ}p(x)\log q(x)=H(p)+D_{KL}(p||q)$

可以看出，交叉熵与相对熵仅相差了H(p)，当p已知时，可以把H(p)看做一个常数，此时交叉熵与KL距离在行为上是等价的，都反映了分布p，q的相似程度。最小化交叉熵等于最小化KL距离。它们都将在p=q时取得最小值H(p)，因为p=q时KL距离为0。可以证明$D_{KL}(p||q)\ge0$，所以$CEH(p,q)\ge H(p)$.

## 交叉熵损失函数

特别考虑神经网络中的分类问题，交叉熵的使用。

*TODO*：怎么从sigmoid或者softmax中使用交叉啥？

### 交叉熵 + sigmoid

二分类问题，真实样本的标签$y_i\in[0,1]$，模型最后通过sigmoid函数，输出为一个概率值。概率越大，代表$y_i=1$的可能性越大。sigmoid公式：$g(s) = \frac{1}{1+e^{-s}}$，其中s是模型上一层的输出，这里g(s)就是模型预测的输出，预测当前样本标签为1的概率$\hat{y} = P(y=1|x)$。

从极大似然的角度出发，$P(y|x) = \hat{y}^y*(1-\hat{y})^{1-y}$. 当真实标签$y=0$，$P(y=0|x) = 1- \hat{y}$；当$y=1$，$P(y=1|x) = \hat{y}$。

变成对数形式，不影响单调性：$\log P(y|x) =y \log\hat{y} +(1-y)\log (1-\hat{y})$.（其负数形式这就是交叉熵，比较真实和预测的距离）希望交叉熵$-\log P(y|x)$越小越好，单个样本i的损失函数为$Loss =-[y_i \log\hat{y}_i +(1-y_i)\log (1-\hat{y}_i)]$

N个样本的损失函数：$Loss =-\frac{1}{N}\sum\limits_{i=1}^N[y_i \log\hat{y}_i +(1-y_i)\log (1-\hat{y}_i)]$

### 交叉熵 + softmax

> 尽管交叉熵刻画的是两个概率分布之间的距离，但是神经网络的输出却不一定是一个概率分布。为此我们常常用Softmax将神经网络前向传播得到的结果变成概率分布。[2]
>
> 在分类问题中用交叉熵可以更好的体现loss的同时，使其仍然是个凸函数，这对于梯度下降时的搜索很有用。反观平方和函数，经过softmax后使得函数是一个非凸函数。[2]
>
> softmax是由逻辑斯的回归模型（用于二分类）推广得到的多项逻辑斯蒂回归模型（用于多分类）[4]

Softmax函数：假设向量$z$是一维向量，长度为K，$p_i = \frac{e^{z_i}}{\sum_{j=1}^Ke^{z_j}}$，即**softmax把向量z变成向量p，p是概率形式，维度没有变化**。

如果使用交叉熵损失函数，那么这一个样本（一维向量）的损失为$L = -\sum\limits_{k=1}^Ky_k\log p_k$.

### 神经网络中的softmax + 交叉熵

考虑多分类问题，样本的label为0-1向量。假设向量z是神经网络最后一层的输出，这一层共有K个神经元，对应K种不同的分类。 假设最后一层的计算公式为$z_i = \sum\limits_{j}w_{ij}x_{ij} + b_i$，对于神经元$i=1,\dots, K$。

现在考虑softmax+交叉熵在神经网络的反向传播。这里只考虑对最后一层的w和b的导数：$\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial z_i}\frac{\partial z_i}{\partial w_i}$和$\frac{\partial L}{\partial b_i} = \frac{\partial L}{\partial z_i}\frac{\partial z_i}{\partial b_i}$.

因为$\frac{\partial z_i}{\partial w_i}=x$，$\frac{\partial z_i}{\partial b_i}=1$，所以只需求$\frac{\partial L}{\partial z_i}$。因为任意的$p_k$均包含$z_i$，由链式法则$\frac{\partial L}{\partial z_i} = \sum\limits_{k=1}^K[\frac{\partial L}{\partial p_k}\frac{\partial p_k}{\partial z_i}]$.其中$\frac{\partial L}{\partial p_k} = \frac{\partial(-\sum_{k=1}^Ky_k\log p_k)}{\partial p_k} = -\frac{y_k}{p_k}$，再求$\frac{\partial p_k}{\partial z_i}$，分为两种情况：

- $k\neq i$

  $$\frac{\partial p_k}{\partial z_i} = \frac{\partial(\frac{e^{z_k}}{\sum_{j=1}^Ke^{z_j}})}{\partial z_i} = \frac{-e^{z_k}e^{z_i}}{(\sum_{j=1}^Ke^{z_j})^2} = -\frac{e^{z_k}}{\sum_{j=1}^Ke^{z_j}}\frac{e^{z_i}}{\sum_{j=1}^Ke^{z_j}} = -p_kp_i$$

- $k=i$

  $ \begin{align} \frac{\partial p_i}{\partial z_i} &= \frac{\partial\left( \frac{e^{z_i}}{\sum_{j=1}^Ke^{z_j}}\right)}{\partial z_i} \\ &= \frac{e^{z_i}\sum_{j=1}^Ke^{z_j} - (e^{z_i})^2}{\left( \sum_{j=1}^Ke^{z_j}\right)^2}  \\ &= \frac{e^{z_i}}{\sum_{j=1}^Ke^{z_j}}\frac{\sum_{j=1}^Ke^{z_j}-e^{z_i}}{\sum_{j=1}^Ke^{z_j}} \\ & = \frac{e^{z_i}}{\sum_{j=1}^Ke^{z_j}}\left(1- \frac{e^{z_i}}{\sum_{j=1}^Ke^{z_j}}\right)  \\ & = p_i(1-p_i)\end{align} $

综上，

$\begin{align}\frac{\partial L}{\partial z_i}&=\sum\limits_{k=1}^K\frac{\partial L}{\partial p_k}\frac{\partial p_k}{\partial z_i}\\ & =\sum\limits_{k=1}^K-\frac{y_k}{p_k}\frac{\partial p_k}{\partial z_i}\\ &=-\frac{y_i}{p_i}p_i(1-p_i)+\sum\limits_{k=1,k\neq i}^K-\frac{y_k}{p_k}-p_kp_i \\ & =y_i(p_i-1)+\sum\limits_{k=1,k\neq i}^K y_kp_i \\ &=p_i\sum\limits_{k=1}^K y_k - y_i \end{align}$

由于是多分类问题，`每个`样本的标签$y=[y_1, y_2, \dots, y_K]$，只会有一个为1，其余均为0，所以$\sum\limits_{k=1}^Ky_k=1$，所以多分类问题，$\frac{\partial L}{\partial z_i}=p_i-y_i$，所以$\frac{\partial L}{\partial w_i}=(p_i-y_i)x_i$，$\frac{\partial L}{\partial b_i}=p_i-y_i$，这里i代表第i个神经元的w和b。**以上只是对一个样本的反向传播**。

## 参考资料

1. [交叉熵损失的来源、说明、求导与pytorch实现](https://zhuanlan.zhihu.com/p/67782576)
2. [交叉熵(Cross Entropy loss)](https://www.cnblogs.com/o-v-o/p/9975365.html)
3. [熵与信息增益](<https://blog.csdn.net/xg123321123/article/details/52864830>)
4. [常用损失函数小结](https://blog.csdn.net/zhangjunp3/article/details/80467350)
5. [超详细的Softmax求导](<https://blog.csdn.net/bqw18744018044/article/details/83120425>)
6. <https://gist.github.com/karpathy/d4dee566867f8291f086>
