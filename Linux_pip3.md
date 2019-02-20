Ubuntu中使用pip3报错

使用pip3 出现以下错误：
```
Traceback (most recent call last): 
File “/usr/bin/pip3”, line 9, in 
from pip import main
```
 

找到 pip3的执行文件

`cd /usr/bin/pip3`

原代码：
```python
1 from pip import main
2 if __name__ == '__main__':
3     sys.exit(main())
```
修改：
```python
1 from pip import __main__    # 将main改成__main__
2 if __name__ == '__main__':
3     sys.exit(__main__._main())  # 将main()改成 _main()
```

保存后再升级pip3：

`pip3 install --upgrade pip`