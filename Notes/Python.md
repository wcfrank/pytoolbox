# Python

在生产环境中，requirements.txt文件里可以包含package的下载网址：

例如，在安装spacy需要的model时（例如en_core_web_sm），因为种种SSL问题安装不上，可以通过直接`pip install URL`的方式安装，在requirements.txt文件中也可以直接写：

```
spacy>=2.0.0,<3.0.0
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz#egg=en_core_web_sm
```
Specifying #egg= with the package name tells pip which package to expect from the download URL. This way, the package won’t be re-downloaded and overwritten if it’s already installed - just like when you’re downloading a package from PyPi.
