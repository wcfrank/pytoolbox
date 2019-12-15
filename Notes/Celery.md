# Celery usage

## 0. 环境搭建

Celery需要使用broker来调度、存储异步队列，这里用的是Redis. 然后再开启celery服务，如果是flask服务，还需再开启flask服务。

- Redis： 下载Redis的Docker镜像，运行`docker run -d -p 6379:6379 redis`，开启Redis

- Celery：在python文件中建立celery实例，并配置使用Redis；最后开启celery服务

  ```python
  CELERY_BROKER_URL = 'redis://redis:6379/0' 
  CELERY_RESULT_BACKEND = 'redis://redis:6379/0'
  ```

- 启动flask服务，然后调用celery装饰过的函数，开始异步方法。

进一步，可使用docker-compose将Celery+Redis+Flask一起打包成服务。

## 1. 基本用法

首先定义两个Celery的函数：

```python
from proj.celery import app # import celery instance

@app.task()
def add(x, y, debug=False):
    if debug:
        print("x: %s; y: %s" % (x, y))
    return x + y

@app.task()
def log(msg):
    return "LOG: %s" % msg
```

普通用法：

```python
res = add.delay(2,3)
print(res.task_id)
print(res.state)
res.get() # Wait until task is ready, and return its result.
```

这里add函数需要输入2个参数x和y，在add.delay()里面输入add所需要的参数，就会发送到celery在后台自动运行（worker执行）。



## 2. 更加正式一点的用法

使用Celery Signature签名（Subtask子任务）。签名的方式支持两种执行方式：直接执行 & worker执行

1. 直接执行

   ```python
   >>> s_add = add.s(3,4)
   >>> s_add()
   # output: 7
   ```

   `s_add`直接调用，即`s_add()`，为直接执行

2. worker执行（这一部分跟上一节**基本用法**效果一样）

   ```python
   >>> s_add = add.s(3,4)
   >>> s_add.delay() # 类似于 add.delay(3,4)
   <AsyncResult: o65468902-56y5-6670008>
   >>> s_add.apply_async()
   <AsyncResult: ri678kl89-780p-876554y>
   
   s_add = add.s(2,2)
   s_add.delay(debug=True) # 依然可以传参数
   <AsyncResult: 1a2c856nv-4gh3-bv9vbn0>
   s_add.apply_async(kwargs={'debug': True}) 
   <AsyncResult: 2kh75b996-agnh-gfjn8v0>
   ```



Celery的签名（signature）其实是偏函数，是固定了部分参数的任务签名

```python
>>> s_add_1 = add.s(1)
>>> s_add_1(10)
11
>>> s_add_1.deplay(20)
<AsyncResult: re3o35435kln4tkj54>
```

**celery.task装饰过的函数，经过signature赋值，直接调用便是`直接执行`，使用delay便是`workder执行`。**

## 3. Celery工作流

### group任务组

任务组函数接收`一组任务签名列表`，返回一个新的任务签名（组签名），调用签名组会执行其包含的所有任务签名，并返回所有结果的列表。

```python
>>> from celery import group
>>> add_group_sig = group(add.s(i, i) for i in range(10))
>>> result = add_group_sig.delay()
>>> result.get()
[0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
# 返回多个结果
>>> result.results
[<AsyncResult: 1716cfd0-e87c-4b3d-a79f-1112958111b1>, 
 <AsyncResult: a7a18bde-726e-49b2-88ed-aeba5d3bf5f2>, 
 <AsyncResult: b9d9c538-2fad-475a-b3d1-bd1488278ce2>, 
 <AsyncResult: 6f370fdd-ed7e-430a-a335-af4650ca15cf>, 
 <AsyncResult: a6ddbe14-5fbd-4079-9f12-35ebbc89d89b>, 
 <AsyncResult: 65dece11-9f38-4940-9fa0-7fcf09266c7a>, 
 <AsyncResult: 8205ffc0-1056-469a-a642-96676d1518e7>, 
 <AsyncResult: e77b7e2b-66d2-48b8-9ffd-4f8fa7d9f4a4>, 
 <AsyncResult: 355b7d01-72c1-4b00-8572-407e751d76c3>, 
 <AsyncResult: aa561ac3-656f-4c81-9e3c-00c64ca49181>] 
```

### chain任务链

任务链函数接收`一组任务签名`，返回一个新的任务签名（链签名），调用签名组会串行执行其包含的所有任务签名，每个任务执行的结果都会作为下一个任务签名的第一个实参，最后只返回一个结果。

```python
>>> from celery import chain
>>> add_chain_sig = chain(add.s(1, 2), add.s(3))
# 精简语法
>>> add_chain_sig = (add.s(1, 2) | add.s(3))
>>> result = add_chain_sig.delay()          # ((1 + 2) + 3)
>>> result.status
u’SUCCESS’
>>> result.get()
6
# 仅返回最终结果
>>> result.results
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'AsyncResult' object has no attribute 'results'
# 结合偏函数
>>> add_chain_sig = chain(add.s(1), add.s(3))
>>> result = add_chain_sig.delay(3)        # ((3 + 1) + 3)
>>> result.get()
7
```

### chord 复合任务

复合任务函数生成一个任务签名时，会先执行一个组签名（不支持链签名），等待任务组全部完成时执行一个回调函数。（回调函数的定义，查看Reference）

```python
>>> from celery import chord, group
>>> add_chord_sig = chord(group(add.s(i, i) for i in range(10)), log.s())
>>> result = add_chord_sig.delay()
>>> result.status
u'SUCCESS'
>>> result.get() # 最后返回回调函数的结果
u'LOG: [0, 2, 4, 6, 8, 10, 12, 16, 14, 18]'
```

### chunks 任务块

任务块函数能够让你将需要处理的大量对象分为分成若干个任务块，如果你有一百万个对象，那么你可以创建 10 个任务块，每个任务块处理十万个对象。有些人可能会担心，分块处理会导致并行性能下降，实际上，由于避免了消息传递的开销，因此反而会大大的提高性能。

```python
>>> add_chunks_sig = add.chunks(zip(range(100), range(100)), 10)
>>> result = add_chunks_sig.delay()
>>> result.get()
[[0, 2, 4, 6, 8, 10, 12, 14, 16, 18], 
 [20, 22, 24, 26, 28, 30, 32, 34, 36, 38], 
 [40, 42, 44, 46, 48, 50, 52, 54, 56, 58], 
 [60, 62, 64, 66, 68, 70, 72, 74, 76, 78], 
 [80, 82, 84, 86, 88, 90, 92, 94, 96, 98], 
 [100, 102, 104, 106, 108, 110, 112, 114, 116, 118], 
 [120, 122, 124, 126, 128, 130, 132, 134, 136, 138], 
 [140, 142, 144, 146, 148, 150, 152, 154, 156, 158], 
 [160, 162, 164, 166, 168, 170, 172, 174, 176, 178], 
 [180, 182, 184, 186, 188, 190, 192, 194, 196, 198]]
```

### map/starmap 任务映射

映射函数，与 Python 函数式编程中的 map 内置函数相似。都是将序列对象中的元素作为实参依次传递给一个特定的函数。map 和 starmap 的区别在于，前者的参数只有一个，后者支持的参数有多个。

```
>>> add.starmap(zip(range(10), range(100)))
[add(*x) for x in [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]]
>>> result = add.starmap(zip(range(10), range(100))).delay()
>>> result.get()
[0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
```



## Reference

1. [分布式任务队列 Celery —— 详解工作流](https://www.cnblogs.com/jmilkfan-fanguiju/p/10589782.html)