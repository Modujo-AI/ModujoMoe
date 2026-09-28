# 开源大模型预训练数据集清单

更新日期：2026-09-15  
适用范围：Modujo 的中英双语文本/代码基础模型预训练（不含图像、音频、纯指令微调和偏好数据）。  
机器可读版本：[pretrain_datasets.csv](./pretrain_datasets.csv)

> “公开可下载”不等于“其中每篇内容都已获得商业训练授权”。聚合数据集的许可证通常只覆盖整理、标注或数据库结构，底层网页、图书、论文和代码仍可能保留原始版权。正式训练前必须复核数据集卡、来源条款、地域法规和模型发布方式。本清单是工程选型资料，不是法律意见。

## 分级标准

| 级别 | 含义 | 使用方式 |
|---|---|---|
| S | 当前任务的首选主干或关键能力数据；质量、规模、可复现性较好 | 优先做消融并进入正式配方 |
| A | 高价值补充，能显著改善领域、语言或知识覆盖 | 清洗、去重后按目标能力混入 |
| B | 有用但已较旧、噪声较高、重叠较多或许可复杂 | 有明确缺口时选用，避免与上游数据重复 |
| C | 原料级、受限或风险较高 | 仅研究、重处理，或不直接纳入发布模型 |

级别衡量的是对本仓库“中英基础模型”的工程价值，不代表学术排名。CSV 中的 `openness` 单独描述使用边界：

- `permissive-ish`：数据集层面相对宽松，但仍须尊重底层内容权利和署名要求。
- `mixed/upstream`：每条数据可能有不同的原始许可，应保存来源与许可元数据。
- `research/restricted`：含非商业、研究用途、定制协议或明显不适合直接商用的限制。
- `public-domain`：仍需按适用法域核对公版状态。

## 对当前 Modujo 配方的建议

当前 50% FineWeb-Edu + 50% FineWeb 2 中文适合验证管线，但正式预训练不建议长期保持只有两个网页来源。建议按 token 而不是文档数配比，并从以下初始区间做小规模消融：

| 数据桶 | 初始比例 | 首选来源 |
|---|---:|---|
| 高质量英文网页/教育 | 30–40% | FineWeb-Edu；DCLM Baseline 作对照 |
| 高质量中文网页 | 30–40% | FineWeb 2 `cmn_Hani` + CCI3-HQ，分别保留来源标签 |
| 代码 | 8–15% | The Stack v2 的许可可接受子集 |
| 数学与 STEM | 8–12% | FineMath + OpenWebMath；先做跨源去重 |
| 百科、论文与公共知识 | 5–10% | Wikipedia、peS2o、PMC OA、公版图书 |
| 其他语言 | 0–5% | FineWeb 2 或 HPLT；仅在确有多语目标时加入 |

配比不是最终答案。应固定验证集，在相同 token 预算下比较中文、英文、代码、数学和通识指标，再调整比例。中文网页源之间、FineMath 与 OpenWebMath、The Pile/Dolma 与其组成源之间会有明显重叠，必须做归一化后的文档级与段落级去重。

## 最优先的候选

1. **FineWeb-Edu**：当前英文主干，教育质量筛选强，1.3T token，ODC-By 1.0；建议保留。
2. **FineWeb 2 中文**：当前中文主干，覆盖新、规模大、处理流程可复现；建议保留并与中文专用语料做消融。
3. **CCI3-HQ**：约 500GB 的高质量中文子集，适合验证是否优于纯 FineWeb 2 中文。
4. **DCLM Baseline**：英文网页主干的强力对照组，适合验证 FineWeb-Edu 单一来源偏差。
5. **FineMath / OpenWebMath**：数学与带 LaTeX 的 STEM 文本，能补足通用网页对形式化内容的损失。
6. **The Stack v2**：代码能力主来源；只选可接受许可证，并保留仓库、路径和许可证元数据。
7. **Wikipedia + peS2o/PMC OA**：补百科、科学写作和可靠知识结构；体量不大但意义高。

## 必做的数据治理

- 固定 dataset revision、文件清单、哈希和抓取日期；延续仓库上级 `sources.json` 的做法。
- 每条样本保存 `source`、`url/repo`、`license`、`timestamp`、`language`、`quality_score` 与删除标志。
- 全局去重应发生在所有来源合并后；另外对公开评测集做污染检测和剔除。
- 清除凭证、私钥、邮箱、电话、身份证号等 PII；代码数据额外扫描 secrets 与生成文件。
- 对中文做简繁、全半角和 Unicode 规范化时保留原文或可逆记录，避免破坏代码、数学式和古文。
- 建立 opt-out/takedown 流程；能够从来源 URL 或仓库定位并重建训练清单。
- 不把许可证写成单一布尔值。数据集许可证、底层内容许可证、访问条款与模型发布限制应分别记录。

## 明确不建议直接使用

- **Books3**：图书版权争议显著；即使可找到镜像，也不应作为默认候选。
- **未过滤的 Common Crawl/WARC**：包含大量垃圾、恶意内容、PII、重复和版权不明内容，只适合作为自建数据管线的原料。
- **任意 GitHub 全量抓取**：公开仓库不等于允许任意再利用；优先使用 The Stack v2 并按原始许可证过滤。
- **benchmark train/test 混入预训练**：会污染评测，尤其是 MMLU、GSM8K、MATH、HumanEval、MBPP、C-Eval、CMMLU 等。
- **聊天/论坛数据的无条件全量混入**：Reddit、Stack Exchange 等内容需遵守平台条款、署名和删除要求，并强化隐私过滤。

## 维护方式

以 CSV 为数据源维护：新增数据集时至少填写级别、类别、语言、规模、许可摘要、开放性、用途、主要风险和官方入口。许可证或访问条件无法确认时写 `VERIFY`，不能凭名称推断。每次正式训练把实际使用的 revision、文件哈希和过滤规则写入训练清单，而不是依赖本页的滚动链接。

## 主要官方参考

- [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [FineWeb 2 dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
- [DCLM repository](https://github.com/mlfoundations/dclm)
- [Dolma documentation](https://allenai.github.io/dolma/)
- [CCI collection](https://huggingface.co/collections/BAAI/cci)
- [The Stack v2 / StarCoder2](https://huggingface.co/blog/starcoder2)
- [FineMath dataset card](https://huggingface.co/datasets/HuggingFaceTB/finemath)
- [OpenWebMath dataset card](https://huggingface.co/datasets/open-web-math/open-web-math)
- [HPLT datasets](https://hplt-project.org/datasets/v2.0)
