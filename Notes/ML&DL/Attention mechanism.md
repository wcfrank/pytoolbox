阅读笔记

# [1] 

This attention layer basically learns a weighting of the input sequence and averages the sequence accordingly to extract the relevant information. 

# [2] 

You can use the final encoded state of a recurrent neural network for prediction. This could lose some useful information encoded in the previous steps of the sequence. In order to keep that information, you can use an average of the encoded states outputted by the RNN. But all of the encoded states of the RNN are equally valuable. Thus, we are using a weighted sum of these encoded states to make our prediction.

![attention](https://i0.wp.com/androidkt.com/wp-content/uploads/2018/11/Attention-RNN-e1543424240331.png?w=722)

# [3] 

Attention机制其实就是**一系列注意力分配系数**，也就是一系列权重参数罢了。

### seq2seq

**注意力模型就是要从序列中学习到每一个元素的重要程度，然后按重要程度将元素合并。**

### 抛开seq2seq

下图是attention函数的本质：一个查询(query)到一系列键值(key-value)对的映射。 

![attention函数的本质](https://pic2.zhimg.com/80/v2-44d2f6f9f60ca21c8b475c12728ae81d_hd.jpg)

一共三步得到attention value：

- 阶段1：Query与Key进行相似度计算得到权值
- 阶段2：对上部权值归一化
- 阶段3：用归一化的权值与Value加权求和

## 参考资料

1. [LSTM With Attention For Relation Classification](https://www.depends-on-the-definition.com/attention-lstm-relation-classification/)

2. [Text Classification using Attention Mechanism in Keras](<http://androidkt.com/text-classification-using-attention-mechanism-in-keras/>)

3. [浅谈Attention机制的理解](<https://zhuanlan.zhihu.com/p/35571412>)