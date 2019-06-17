# Vanilla RNN

用numpy写一个最简单的RNN模型（by [gist of Andrej Karpathy](<https://gist.github.com/karpathy/d4dee566867f8291f086>)）。深入研究Karpathy的代码。

```python
import numpy as np
data = open('input.txt', 'r').read() # should be simple plain text file. A list of strings
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
```

输入是本文，如一本小说。data应该是一个list，每个元素是一个单词。char_to_ix是把单词映射成数字，ix_to_char是把数字映射成单词。

# hyperparameters
```python
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1
```

模型的参数：

`Wxh`：

`Whh`：

`Why`：

```python
# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias
```

