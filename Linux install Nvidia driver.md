目的：在Ubuntu主机下安装tensorflow-gpu

根据tensorflow官网的信息，需要进行一系列配置：

1. 安装Nvidia驱动
1. 安装toolkit
1. ...

在安装Nvidia驱动的时候遇到问题：

首先从Nvidia官网下载好了对应的驱动.run文件，但是直接双击安装一直不成功。
上网找资料发现不能直接安装

[ubuntu16.04系统run方式安装nvidia显卡驱动](https://blog.csdn.net/xunan003/article/details/81665835)

然后就按照这篇博客的说明，

1. 先禁用自带的nouveau驱动
2. 进入Ubuntu的字符界面 
    `sudo service lightdm stop      //这个是关闭图形界面，不执行会出错`
    `sudo apt-get remove nvidia-*  `
3. `sudo chmod  a+x NVIDIA-Linux-x86_64-396.18.run`
    安装过程中的选项：（这是copy别人的，自己的没记住，我也是尝试选择了好多遍才安装好）
    The distribution-provided pre-install script failed! Are you sure you want to continue? 选择 yes 继续。
    Would you like to register the kernel module souces with DKMS? This will allow DKMS to automatically build a new module, if you install a different kernel later?  选择 No 继续。
    问题没记住，选项是：install without signing
    问题大概是：Nvidia's 32-bit compatibility libraries? 选择 No 继续。
    Would you like to run the nvidia-xconfigutility to automatically update your x configuration so that the NVIDIA x driver will be used when you restart x? Any pre-existing x confile will be backed up.  选择 Yes  继续
    这些选项如果选择错误可能会导致安装失败，没关系，只要前面不出错，多尝试几次就好。
4. 挂载Nvidia驱动：`modprobe nvidia`
5. 检查驱动是否安装成功：`nvdia-smi`

在第2步时遇到问题：无法通过Ctrl+Alt+F1进入字符界面，后来发现这篇博客：

[Ubuntu 16.04纯文本界面、图形化界面切换方法](https://blog.csdn.net/davidhopper/article/details/79288573)
