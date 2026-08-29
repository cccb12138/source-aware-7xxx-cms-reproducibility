# GitHub 仓库整理与发布详细步骤

## 一、本地工作副本

本次发布使用新的干净仓库：

```powershell
cd F:\CC\github_source-aware-7xxx-cms-reproducibility
git branch --show-current
git status --short
```

预期分支为 `main`。新仓库不导入旧仓库提交历史，也不创建旧的 `v1.0.0` tag。旧仓库和旧 Zenodo 1.0.0 仅作为历史版本保留。

## 二、文件进入仓库的规则

1. `code/frozen_original/scripts/` 保存原始 00–48 分析链，不在其中直接改路径。
2. `requirements.txt`、`requirements-lock.txt` 和 `environment.yml` 保存已核对版本。
3. `data/folds/` 保存固定 source-group folds，不重新随机生成。
4. `data/source_index/` 保存来源索引和重建说明。
5. `data/public/` 只允许放入许可审查状态为 `Approved_open` 的记录。
6. `results/summary/` 只放汇总结果，不放 `UTS_final_oof_predictions.csv`。
7. `figures/` 使用 CMS 最终 Fig.1–Fig.7 编号；Fig.4 必须显示 0.527。
8. 不上传总工作簿、参考 PDF、工作簿预览、本机缓存、OneDrive 路径或任何凭据。

## 三、本地验证

```powershell
python code\validate_release.py
git status --short
git diff --stat
git diff --check
```

逐项查看变更：

```powershell
git diff -- README.md CITATION.cff DATA_LICENSES.md
git diff -- data\folds data\source_index
git diff -- results\summary figure_data
```

验证通过后暂存：

```powershell
git add README.md CITATION.cff LICENSE DATA_LICENSES.md CHANGELOG.md
git add requirements.txt requirements-lock.txt environment.yml
git add code data docs results figure_data figures manifests config .gitignore
git diff --cached --stat
git diff --cached --check
```

提交前再次确认 staged files 中没有 row-level cohort、OOF predictions、Excel 总表或参考 PDF。

## 四、初次提交和推送新仓库

```powershell
git commit -m "Initial CMS reproducibility repository"
git push -u origin main
```

确认 GitHub 网页能够直接看到 README、frozen scripts、环境、folds、source index、正确 CMS 图表和许可。由于来源许可审查未完成，仓库不得声称已经发布完整 row-level 数据。

## 五、启用 Zenodo–GitHub 自动归档

在 Zenodo 的 GitHub 设置中关闭旧仓库连接，并为新仓库 `cccb12138/source-aware-7xxx-cms-reproducibility` 打开开关。确认新仓库显示在 `Enabled Repositories` 中且状态为 `ON`。

自动归档和手工上传只选择一种。本仓库采用 Zenodo–GitHub 自动归档：Zenodo 会在 GitHub release 发布后下载该 release 的源码归档并生成新的 DOI。不要同时新建手工 Zenodo 上传，以免得到重复记录。

GitHub release 发布前，`CITATION.cff` 必须已经写成正式 `1.1.0`、正确发布日期和新仓库 URL。DOI 字段暂时省略，等 Zenodo 自动生成后再同步。

## 六、打最终 tag

只有在 GitHub `main`、CITATION、README、本地验证结果和预期 release 文件完全一致后执行：

```powershell
git switch main
git pull --ff-only origin main
git tag -a v1.1.0 -m "CMS reproducibility release v1.1.0"
git push origin v1.1.0
```

新仓库不创建旧 `v1.0.0` tag。`v1.1.0` 必须指向最终审核过的 commit，发布后不要移动。

## 七、创建 GitHub release

在 GitHub Releases 页面选择 `Draft a new release`，选择现有 tag `v1.1.0`，标题建议为 `CMS reproducibility release v1.1.0`。release notes 至少说明：

- 对应 CMS 投稿稿件；
- frozen script 范围和哈希；
- 固定环境与 folds；
- 公布的数据类型和因许可未公布的数据；
- Fig.4 已从旧 0.547 标注修正为最终 0.527；
- 与 v1.0.0 的差异。

发布前下载 release 自动生成的 source archive，并在临时目录再次运行 `python code/validate_release.py`。GitHub release 由 tag 指定的 commit 生成，并自动提供 ZIP 和 tar.gz 源码归档。

## 八、核对 Zenodo 自动归档

点击 GitHub 的 `Publish release` 后等待 Zenodo 自动归档。不要另行上传 ZIP。新记录出现后确认：

1. Zenodo 文件来自 GitHub `v1.1.0` release；
2. creator 为 Bo Chang；
3. title 与仓库/CITATION 一致；
4. version 为 1.1.0；
5. related identifier 指向 GitHub release；
6. access-rights 和数据许可表述与仓库一致。

Zenodo 已为 `v1.1.0` 生成 DOI：<https://doi.org/10.5281/zenodo.22162478>。该 DOI 已同步加入 `CITATION.cff`、README 和论文。这个 DOI 同步提交发生在 `v1.1.0` tag 之后，因此不能移动或重写已经发布的 tag。论文中引用新 DOI，不继续把旧 DOI 描述成最终投稿代码。

## 九、论文 Data availability 最终核对

逐句检查：GitHub release/tag 存在；Zenodo DOI 可解析；两个归档的文件一致；代码、folds、source index 和汇总结果确实存在；全文没有声称公开尚未发布的 row-level 数据。

## 官方操作参考

- GitHub release：<https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository>
- Zenodo–GitHub 集成：<https://help.zenodo.org/docs/github/>
