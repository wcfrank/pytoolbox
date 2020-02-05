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
<!--
```shell
$:> git diff --color-words
diff --git a/diff_test.txt b/diff_test.txt
index 6b0c6cf..b37e70a 100644
--- a/diff_test.txt
+++ b/diff_test.txt
@@ -1 +1 @@
this is a git difftest example
```
-->
### （重要）搞清楚git diff, git diff --cached和git diff HEAD
Git官网上的解释：
- git diff [<options>] [--] [<path>…]
  This form is to view the changes you made relative to the index (staging area for the next commit). In other words, the differences are what you could tell Git to further add to the index but you still haven’t. You can stage these changes by using `git-add`.
  默认情况下， `git diff` 会显示所有没有被`git add`的修改。
- git diff [<options>] --cached [<commit>] [--] [<path>…]
  This form is to view the changes you staged for the next commit relative to the named <commit>. Typically you would want comparison with the latest commit, so if you do not give <commit>, it defaults to HEAD. --staged is a synonym of --cached.
  `git diff --cached`会显示已经`git add`但没有`git commit`的内容，跟HEAD（也可以是其他commits）比较的修改。
- git diff [<options>] <commit> [--] [<path>…]
  This form is to view the changes you have in your working tree relative to the named <commit>. You can use HEAD to compare it with the latest commit, or a branch name to compare with the tip of a different branch.
  `git diff HEAD`会显示工作区与HEAD比较的修改。

一般时候，在git仓库随手做了一些代码改动，但并不想加入git add或commit，可通过`git diff`查看都做了哪些改动。（`git diff HEAD`也可以）


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
## 4. git log 显示已提交的快照(只作用于已提交的历史）
- `git log -n 3` 只显示3个提交
- `git log --oneline`将每个提交压缩到一行
- `git log --stat` 除了 git log 信息之外，包含哪些文件被更改了，以及每个文件相对的增删行数
- `git log -p` 显示代表每个提交的一堆信息。显示每个提交全部的差异（diff），这也是项目历史中最详细的视图
- `git log --author="<pattern>"` 搜索特定作者的提交。<pattern> 可以是字符串或正则表达式
- `git log --grep="<pattern>"` 搜索提交信息匹配特定 <pattern> 的提交。<pattern> 可以是字符串或正则表达式
- `git log <since>..<until>` 只显示发生在 <since> 和 <until> 之间的提交。两个参数可以是提交 ID、分支名、HEAD 或是任何一种引用
- **`git log <file>`** 只显示包含特定文件的提交。查找特定文件的历史这样做会很方便
- `git log --graph --decorate --oneline` 还有一些有用的选项: --graph 标记会绘制一幅字符组成的图形，左边是提交，右边是提交信息；--decorate 标记会加上提交所在的分支名称和标签；--oneline 标记将提交信息显示在同一行，一目了然

#### 例子
1. `git log --author="John Smith" -p hello.py` 这个命令会显示 John Smith 作者对 hello.py 文件所做的所有更改的差异比较（diff）
2. `git log --oneline master..some-feature` ..句法是比较分支很有用的工具，这条命令显示了在 some-feature 分支而不在 master 分支的所有提交的概览

## 5. git revert
git revert 命令用来撤销一个已经提交的快照。但是，它是通过搞清楚如何撤销这个提交引入的更改，
然后在最后加上一个撤销了更改的*新*提交，而不是从项目历史中移除这个提交。
下面的这个栗子是 git revert 一个简单的演示。它提交了一个快照，然后立即撤销这个操作。

```shell
# 编辑一些跟踪的文件
# 提交一份快照
git commit -m "Make some changes that will be undone"
# 撤销刚刚的提交
git revert HEAD
```

## 6. git commit --amend 修复最新提交
将缓存的修改和之前的提交合并到一起，而不是提交一个全新的快照。它还可以用来简单地编辑上一次提交的信息而不改变快照。
amend 不只是修改了最新的提交——它进行了一次替换。
合并缓存的修改和上一次的提交，用新的快照替换上一个提交。
缓存区没有文件时运行这个命令可以用来编辑上次提交的提交信息，而不会更改快照。

**但不要修复公共提交**

仓促的提交在你日常开发过程中时常会发生。很容易就忘记了缓存一个文件或者弄错了提交信息的格式。**--amend标记是修复这些小意外的便捷方式。**
```shell
# 编辑 hello.py 和 main.py
git add hello.py
git commit

# 意识到你忘记添加 main.py 的更改
git add main.py
git commit --amend --no-edit
```
编辑器会弹出上一次提交的信息，加入 --no-edit 标记会修复提交但不修改提交信息。
完整的提交会替换之前不完整的提交，看上去就像我们在同一个快照中提交了 hello.py 和 main.py。
