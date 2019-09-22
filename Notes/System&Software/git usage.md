# Git的用法
摘自https://github.com/geeeeeeeeek/git-recipes
<br>
</br>


## 1. 分拆同一文件的修改内容，add到不同的缓存顺序
```
git add -p
```

开始交互式的缓存，你可以将某个文件的其中一处更改加入到下次提交缓存。
Git 会显示一堆更改，并等待用户命令。
使用 `y` 缓存某一处更改，使用 `n` 忽略某一处更改，使用 `s` 将某一处分割成更小的几份，使用 `e` 手动编辑某一处更改，使用  `q` 退出编辑。

## 2. 使用 git diff 对比更改

`diff` 命令读入两个数据集，并输出两者之间的更改。
`git diff` 是一个用途广泛的 Git 命令，对 Git 中的数据进行 `diff` 差异比较。它接受的参数可以是提交、分支和文件等数据。
本文将会介绍 `git diff` 的常见使用场景和差异差异比较的工作流。
`git diff` 命令通常和 `git status`和 `git log` 一同使用来分析 Git 仓库当前的状态。

```shell
diff --git a/diff_test.txt b/diff_test.txt
```

```shell
$:> git diff --color-words
diff --git a/diff_test.txt b/diff_test.txt
index 6b0c6cf..b37e70a 100644
--- a/diff_test.txt
+++ b/diff_test.txt
@@ -1 +1 @@
this is a git difftest example
```
### 上次提交之后的更改

默认情况下， `git diff` 会显示上次提交后所有未提交的更改。

### 对比两个分支

分支的比较与其他传给 `git diff` 的引用相同：

```shell
git diff branch1..other-feature-branch
```

这个栗子引入了“点”操作符。其中两个点表示 diff 的输入是两个分支的顶端。当你用空格替代这两个点时，它们的效果相同。另外还有三点操作符：

```shell
git diff branch1...other-feature-branch
```

三点操作符首先将第一个输入参数 `branch1` 修改成两个 diff 输入 branch1 分支和 other-feature-branch 分支的共同公共祖先的引用。
最后一个输入参数保留不变，为 other-feature-branch 分支的顶端。

### 对比两个分支中的文件

将文件名作为第三个参数传入 `git diff` 命令，以比较不同分支上的同一文件：

```shell
git diff master new_branch ./diff_test.txt
```
```shell
git diff
```

### 两次提交之间的更改

`git diff` 可以将 Git 提交引用传给 diff 命令。例如，引用包括 `HEAD`、标签和分支名称。
每个 Git 中的提交都有一个提交编号，你就有执行 `git log` 命令获得。你也可以将这个编号传给 `git diff`。

```shell
$:> git log --prety=oneline
957fbc92b123030c389bf8b4b874522bdf2db72c add feature
ce489262a1ee34340440e55a0b99ea6918e19e7a rename some classes
6b539f280d8b0ec4874671bae9c6bed80b788006 refactor some code for feature
646e7863348a427e1ed9163a9a96fa759112f102 add some copy to body
$:> git diff 957fbc92b123030c389bf8b4b874522bdf2db72c ce489262a1ee34340440e55a0b99ea6918e19e7a
```

### 对比两个分支中的文件

将文件名作为第三个参数传入 `git diff` 命令，以比较不同分支上的同一文件：

```shell
git diff master new_branch ./diff_test.txt
```

## 3. git stash
