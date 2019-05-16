# Keras中Masking的作用

## Masking层的作用[2]

使用给定的值，对输入序列信号进行屏蔽，用以定位需要跳过的时间步。对于输入张量维度为(samples, timestep, features)，某个时间步上feature的值都等于mask_value，则该时间步在模型接下来所有层被跳过。

如果模型接下来有层不支持masking，却接受到masking过的数据，则抛出异常。

Embedding层也有过滤的功能，但只能过滤0，不能指定其他字符。

## 例1：自定义实现带masking的averagepooling层

使用LSTM的时候，样本的长度不一样。对不定长序列的一种预处理方法是，首先对数据进行padding补0，然后引入keras的Masking层，它能自动对0值进行过滤。 问题在于keras的某些层不支持Masking层处理过的输入数据，例如Flatten、AveragePooling1D等等。例如LSTM对每一个序列的输出长度都等于该序列的长度，那么均值运算就只应该除以序列本身的长度，而不是padding后的最长长度。[1]

### 自定义keras层

自定义实现带masking功能的池化层。keras中自定义层需要实现三个方法：

- `buid(input_shape)`：定义参数
- `call(x)`：编写层的功能逻辑；如果希望支持masking，需要另外传入masking参数
- `compute_output_shape(input_shape)`：如果改变了输入的形状，需要在这里定义shape的变化

### 实例

```python
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Input, Masking
from keras.models import Model
import tensorflow as tf
import numpy as np

class MyMeanPool(Layer):
    def __init__(self, axis, **kwargs):
        self.supports_masking = True # important
        self.axis = axis
        super(MyMeanPool, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None

    def call(self, x, mask=None):
        if mask is not None:
            mask = K.repeat(mask, x.shape[-1]) 
            mask = tf.transpose(mask, [0,2,1])
            mask = K.cast(mask, K.floatx())
            x = x * mask
            return K.sum(x, axis=self.axis) / K.sum(mask, axis=self.axis)
        else:
            return K.mean(x, axis=self.axis)

    def compute_output_shape(self, input_shape):
        output_shape = []
        for i in range(len(input_shape)):
            if i!=self.axis:
                output_shape.append(input_shape[i])
        return tuple(output_shape)

data = [[[10,10],[0, 0 ],[0, 0 ],[0, 0 ]],
        [[10,10],[20,20],[0, 0 ],[0, 0 ]],
        [[10,10],[20,20],[30,30],[0, 0 ]],
        [[10,10],[20,20],[30,30],[40,40]]] # size: 4*4*2
data = np.array(data)

A = Input(shape=[4,2]) # None*4*2
mA = Masking()(A)
out = MyMeanPool(axis=1)(mA)
model = Model(inputs=[A], outputs=[out])
print(model.predict(data)) # 没有build方法，参数不需要训练，直接预测
# output: [[10. 10.], [15. 15.], [20. 20.], [25. 25.]]
```

现在明白了masking的作用了，但是在这个例子里面，`call`方法传入的`mask`参数是什么样子？它怎么作用在`x`上面呢？需要进一步研究一下，我对上面的代码稍作修改，让`call`方法直接输入`mask`参数，查看它长什么样子：

```python
class MyMeanPool_2(MyMeanPool): # 继承前一个类，只修改call方法
    def call(self, x, mask=None):
		return mask
    
A = Input(shape=[4,2]) # None * 4 * 2
mA = Masking()(A)
out = MyMeanPool_2(axis=1)(mA)
model = Model(inputs=[A], outputs=[out])
print(model.predict(data))
# outputs: array([[ True, False, False, False], [ True,  True, False, False], [ True,  True,  True, False], [ True,  True,  True,  True], [ True,  True,  True,  True]])
```

原来，上一层的Masking层，传过来的`mask`参数是一个boolean的array，对每一个样本的每一个timestep，对应一个True或者False值，表示是否这个timestep的feature value全部满足mask_value。

这样就好理解了，第一个实例`MyMeanPool`类的`call`方法：

1. 先通过`repeat`把`mask`复制2倍（因为`x.shape[-1]=2`）得到`mask`的shape为(None, 2, 4)。第一条样本变为：`[[True, False, False, False], [True, False, False, False]]`
2. 通过转置得到shape为(None, 4, 2)。第一条样本变为：`[[True, True], [False, False], [False, False], [False, False]]`
3. 通过`cast`方法将Boolean值转成数值。第一条样本变为：`[[1., 1.], [0., 0.], [0., 0.], [0., 0.]]`
4. `x = x * mask`是出于保险起见，不加也问题不大



## 例2：自定义实现带masking的flatten层

[3]自带的Flatten层不支持masking，自定义实现。**mask总是与input具有相同的shape**，所以需要在`compute_mask`方法里对mask也做flatten。（存疑）

```python
# 给定输入：
import numpy as np
data2 = [[1,0,0,0],
        [1,2,0,0],
        [1,2,3,0],
        [1,2,3,4]]
data2 = np.array(data2) # size (4, 4)
```

期待的输出为：`[3., 6., 9., 12.]`，即非0的元素数*3。接下来代码想加入flatten层，所以最后再加一层求和层。

```python
class MyFlatten(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MyFlatten, self).__init__(**kwargs)

    def compute_mask(self, inputs, mask=None):
        if mask==None:
            return mask
        return K.batch_flatten(mask)

    def call(self, inputs, mask=None):
        return K.batch_flatten(inputs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], np.prod(input_shape[1:]))
```

[3]中只给出了`Myflatten`层的定义，没有给出`MySum`层定义，自己尝试写一下

### 方法一：

1. 将mask从(None, 4)转变为(None, 4, 3)
2. 将inputs从(None, 12)转变为(None, 4, 3)
3. mask和inputs按照元素相乘
4. 相加

```python
class MySum(Layer):
    def __init__(self, axis=1, **kwargs):
        self.support_masking = True
        self.axis = axis
        super(MySum, self).__init__(**kwargs)
    
    def compute_mask(self, inputs, mask=None):
        return None
    
    def call(self, inputs, mask=None): 
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.repeat(mask, 3)
            mask = tf.transpose(mask, [0,2,1])
            
            inputs = K.reshape(inputs, shape=(-1,4,3))
            inputs = inputs * mask
            inputs = K.sum(inputs, axis=1) 
            return K.sum(inputs, axis=1)
        else:
            return K.sum(inputs, axis=self.axis)
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)
    
from keras.initializers import Ones

A = Input(shape=[4]) # None * 4
emb = Embedding(5, 3, mask_zero=True, embeddings_initializer=Ones())(A) # None * 4 * 3
fla = MyFlatten()(emb)
out = MySum(axis=1)(fla)
model = Model(inputs=[A], outputs=[out])
model.predict(data2)
# outputs: array([ 3.,  6.,  9., 12.], dtype=float32)
```

`call`方法传入的mask参数为`[[Ture, False, False, False], [Ture, Ture, False, False], [Ture, Ture, Ture, False], [Ture, Ture, Ture, Ture]]`，size为(4,4)。（为什么经过上一层的`compute_mask`方法，没有把mask也flatten呢？）经过变形，mask的size最终为(None, 4,3)

inputs参数经过上一层的flatten，已经变成(None, 12)，需要reshape变成(None, 4, 3)

### 方法二：

只把mask从(None, 4)变成(None, 12)

```python
class MySum(Layer):
    def __init__(self, axis=1, **kwargs):
        self.support_masking = True
        self.axis = axis
        super(MySum, self).__init__(**kwargs)
    
    def compute_mask(self, x, mask=None):
        return None
    
    def call(self, x, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.repeat(mask, 3)
            mask = K.batch_flatten(mask)
            x = x * mask
            return K.sum(x, axis=self.axis)
        else:
            return K.sum(x, axis=self.axis)
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)
```

repeat mask参数，然后flatten成为(None, 12)，得到同样的结果。

## 疑问

1. `compute_mask`方法不能自动的传给下一层。实例2传到Mysum层的mask依然是(None, 4)大小？（也可能mask本来就是(None，4)大小。可以通过将MyFlatten层的`compute_mask`方法去掉来验证一下区别！）
2. 实例2中`call`方法的输入inputs无法识别shape

## 参考资料

1. [Keras自定义实现带masking的meanpooling层](https://blog.csdn.net/songbinxu/article/details/80148856)
2. [Keras使用过程中的tricks和errors(持续更新)](<https://zhuanlan.zhihu.com/p/34771270>)
3. [Keras实现支持masking的Flatten层](https://blog.csdn.net/songbinxu/article/details/80254122)