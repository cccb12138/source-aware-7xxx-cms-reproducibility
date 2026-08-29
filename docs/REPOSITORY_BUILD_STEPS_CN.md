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

## 五、先建立 Zenodo 新版本草稿并预留 DOI

打开已发布的 Zenodo 1.0.0 记录，点击 `New version` 建立 1.1.0 草稿。不要修改或替换旧版本文件。在新版本草稿中点击 `Get a DOI now!` 预留新的 version-specific DOI，但暂时不要发布。

将预留 DOI 写入 `CITATION.cff` 和 README，把 `1.1.0-rc1` 改为 `1.1.0`，并填写最终 release date。重新运行验证，然后提交并推送这个 DOI 同步修改。

Zenodo 官方说明：新版本是一个具有独立文件、metadata 和持久标识符的新记录，并与旧版本相互关联；预留 DOI 可以在发布前写入待上传文件。

## 六、合并后打最终 tag

只有在 GitHub `main`、预留 DOI、CITATION、README、本地验证结果和预期 release 文件完全一致后执行：

```powershell
git switch main
git pull --ff-only origin main
git tag -a v1.1.0 -m "CMS reproducibility release v1.1.0"
git push origin v1.1.0
```

不要移动已有 `v1.0.0`。tag 必须指向最终审核过的 commit。

## 七、创建 GitHub release

在 GitHub Releases 页面选择 `Draft a new release`，选择现有 tag `v1.1.0`，标题建议为 `CMS reproducibility release v1.1.0`。release notes 至少说明：

- 对应 CMS 投稿稿件；
- frozen script 范围和哈希；
-固定环境与 folds；
- 公布的数据类型和因许可未公布的数据；
- Fig.4 已从旧 0.547 标注修正为最终 0.527；
- 与 v1.0.0 的差异。

发布前下载 release 自动生成的 source archive，并在临时目录再次运行 `python code/validate_release.py`。GitHub release 由 tag 指定的 commit 生成，并自动提供 ZIP 和 tar.gz 源码归档。

## 八、发布 Zenodo 1.1.0

将 GitHub `v1.1.0` 对应的源码 ZIP 上传到已经预留 DOI 的 Zenodo 1.1.0 草稿。若选择 Zenodo–GitHub 自动归档，则不要再手工建立另一个重复记录；两种方式只选一种。本项目已经存在 1.0.0 记录，采用“New version 草稿 + 预留 DOI + 上传最终 tag ZIP”的手工方式更容易保证精确对应。

发布前确认：

1. Zenodo 文件与 GitHub tag 指向同一 payload；
2. creator 为 Bo Chang；
3. title 与仓库/CITATION 一致；
4. version 为 1.1.0；
5. related identifier 指向 GitHub release；
6. access-rights 和数据许可表述与仓库一致。

确认 Zenodo 草稿中的 DOI 与仓库 `CITATION.cff` 相同、ZIP 校验值与最终 tag archive 相同后再点击 Publish。论文中引用这个新的 version-specific DOI，不继续把旧 v1.0.0 描述成最终投稿代码。

## 九、论文 Data availability 最终核对

逐句检查：GitHub release/tag 存在；Zenodo DOI 可解析；两个归档的文件一致；代码、folds、source index 和汇总结果确实存在；全文没有声称公开尚未发布的 row-level 数据。

## 官方操作参考

- GitHub release：<https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository>
- Zenodo 新版本：<https://help.zenodo.org/docs/deposit/manage-versions/>
- Zenodo 预留 DOI：<https://help.zenodo.org/docs/deposit/describe-records/reserve-doi/>
