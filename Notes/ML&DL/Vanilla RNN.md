# Vanilla RNN

用numpy写一个最简单的RNN模型（by [gist of Andrej Karpathy](<https://gist.github.com/karpathy/d4dee566867f8291f086>)）。深入研究Karpathy的代码。



```python
import numpy as np
data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
```



输入是本文，如一本小说，

