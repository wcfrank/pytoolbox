# Samples with Unbalanced Label

from [1]

在两类比例非常不均衡的情况下，就不能再用「分类正确率」（accuracy）来衡量模型性能，而要用少数类的「准确率」（precision）和「召回率」（recall），或者二者的综合（F1, equal error rate, AUC等等）。

在训练时，如果能够把这些评价指标作为目标函数，那是最好的。

如果你的模型只支持用分类正确率作为目标函数，那么可以有下面几种对策：

1. (weight) 调整两类训练样本的权重，使得两类的总权重相等。这是最省事的办法。
2. (upsampling) 如果你的模型不支持类加权或样本加权，那么可以把少数类的数据复制几份，使得两类数据量基本相同。这与加权重是等效的，但是会浪费空间和时间。
3. (upsampling) 把少数类的数据复制几份，并适当加噪声。这可以增强模型的鲁棒性。
4. (data augmentation) 加噪声可以拓展到一般的 data augmentation —— 根据已有数据生成新的数据，但保证类别相同。data augmentation 的方法有很多，不同性质的数据，augmentation 的方法也不一样。例如，图像数据可以平移、放缩、旋转、翻转等等。
5. (downsampling) 如果多数类的数据量太大，也可以从中随机地取一小部分。当然，此方法一般要与上面 1~4 结合使用。



from [2]，翻译Quora

- 考虑对各类别尝试不同的采样比例，比一定是1:1，有时候1:1反而不好，因为与现实情况相差甚远。
- 决策树往往在类别不均衡数据上表现不错。它使用基于类变量的划分规则去创建分类树，因此可以强制地将不同类别的样本分开。目前流行的决策树算法有：C4.5、C5.0、CART和Random Forest等。
- 使用相同的分类算法，但是使用一个不同的角度，比如你的分类任务是识别那些小类，那么可以对分类器的小类样本数据增加权值，降低大类样本的权值（这种方法其实是产生了新的数据分布，即产生了新的数据集，译者注），从而使得分类器将重点集中在小类样本身上。一个具体做法就是，在训练分类器时，若分类器将小类样本分错时额外增加分类器一个小类样本分错代价，这个额外的代价可以使得分类器更加“关心”小类样本。如penalized-SVM和penalized-LDA算法
- 一个很好的方法去处理非平衡数据问题，并且在理论上证明了。这个方法便是由Robert E. Schapire于1990年在Machine Learning提出的”The strength of weak learnability” ，该方法是一个boosting算法，它递归地训练三个弱学习器，然后将这三个弱学习器结合起形成一个强的学习器。我们可以使用这个算法的第一步去解决数据不平衡问题。 
- 对小类中的样本进行复制以增加该类中的样本数，但是可能会增加bias。
- AUC是最好的评价指标。







参考资料：

1. [机器学习中非均衡数据集的处理方法？](https://www.zhihu.com/question/30492527)（王赟 Maigo的回答）

2. [在分类中如何处理训练集中不平衡问题](https://blog.csdn.net/heyongluoyao8/article/details/49408131)