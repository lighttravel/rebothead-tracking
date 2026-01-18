1、git分支
Git Flow是一个更结构化的工作流程，它定义了多种类型的分支。主分支（main）只包含正式发布的版本，开发分支（develop）作为集成分支，功能分支（feature）用于开发新功能，发布分支（release）用于准备新版本发布，热修复分支（hotfix）用于紧急修复生产环境的bug。这种流程适合中大型项目的团队协作。
创建新分支可以使用git branch 分支名命令，这个命令只会创建分支，不会自动切换到新分支。如果你想创建并立即切换到新分支，可以使用git checkout -b 分支名或git switch -c 分支名。切换分支使用git checkout 分支名或git switch 分支名。
查看所有分支可以使用git branch命令，带有星号标记的是当前所在分支。删除分支使用git branch -d 分支名，但如果分支还有未合并的更改，则需要使用git branch -D 分支名强制删除。合并分支时，先切换到目标分支，然后使用git merge 要合并的分支名命令。
2、
(1)将本地文件提交到test分支
# 1. 查看当前状态（会显示哪些文件有修改）
git status
# 2. 添加所有修改的文件到暂存区
git add .
# 3. 提交到test分支
git commit -m "描述你的修改内容"
# 4. 如果需要推送到远程仓库
git push origin test
(2)将main分支的更改同步到test分支
# 方法1：先切换到main分支拉取最新代码，再切换回test分支合并
git checkout main
git pull origin main
git checkout test
git merge main
git push origin test
# 方法2：在test分支上直接合并main分支
git checkout test
git merge main
git push origin test
(3)从远程仓库同步代码到test分支
# 拉取远程test分支的最新代码
git pull origin test
