# GPT、Veri 与 DeepSeek 共同样本评估报告

**报告日期：** 2026-09-16  
**数据截点：** 2026-09-15 23:46 UTC  
**裁判模型：** GPT-4o-mini  
**指标阈值：** 0.7  
**主比较集：** 三模型共同成功完成的 419 道同题用例  
**结论性质：** 仅针对共同成功评估样本

## 一、结论摘要

1. **DeepSeek 的回答/拒答决策为 100%。** 在 419 题共同样本中，其状态为 AA 194、NN 225、AN 0、NA 0。
2. **Veri 的答案正确性明显更高。** 在共同样本的有答案题中，Veri Correct Answer Rate 为 68.56%，高于 DeepSeek 的 39.18% 和 GPT 的 35.57%；AA Correctness 均值也以 Veri 的 0.749 最高。
3. **GPT 与 DeepSeek 的端到端门控通过率较接近。** GPT 为 58.47%，DeepSeek 为 56.56%，Veri 为 12.89%。Veri 较低的门控通过率不等同于决策能力差，其 Decision Accuracy 仍为 97.37%。
4. **三组结果全部由 GPT-4o-mini 裁判。** 这份报告不代表 DeepSeek 裁判结果；GPT 同时作为考生和裁判，可能存在同模型偏好。

## 二、比较口径

主比较集只保留三个模型均为 `status=completed` 的相同 `(document, name)`，共 419 题：

- 有答案题：194
- 无答案题：225
- `input` 不一致：0
- `expected_answered` 不一致：0
- `expected_output` 不一致：0
- `retrieval_context` 不一致：0

决策状态定义：

| 状态 | 含义 |
| --- | --- |
| AA | 应回答且实际回答 |
| NN | 应拒答且实际拒答 |
| AN | 应回答但实际拒答 |
| NA | 应拒答但实际回答 |

关键公式：

- Decision Accuracy = `(AA + NN) / Total`
- 错误拒答率 = `AN / (AA + AN)`
- NN 决策成功率 = `NN / (NN + NA)`
- 无依据作答率 = `NA / (NN + NA)`
- Correct Answer Rate = `AA 且 Correctness 达标 / 全部有答案题`

## 三、共同样本决策表现

| 模型 | AA | NN | AN | NA | Decision Accuracy | 错误拒答率 | NN 成功率 | 无依据作答率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-4o-mini | 193 | 223 | 1 | 2 | 99.28% | 0.52% | 99.11% | 0.89% |
| Veri | 184 | 224 | 10 | 1 | 97.37% | 5.15% | 99.56% | 0.44% |
| DeepSeek | 194 | 225 | 0 | 0 | **100.00%** | **0.00%** | **100.00%** | **0.00%** |

| 模型 | Answer Precision | Answer Recall | Abstention Precision |
| --- | ---: | ---: | ---: |
| GPT-4o-mini | 98.97% | 99.48% | **99.55%** |
| Veri | 99.46% | 94.85% | 95.73% |
| DeepSeek | **100.00%** | **100.00%** | **100.00%** |

DeepSeek 在当前交集没有发生回答/拒答错误。Veri 的主要决策损失来自 10 个 AN，即材料有答案但系统拒答。

## 四、共同样本回答质量

本节只统计 AA，即材料有答案且模型实际回答的用例。

### 4.1 指标均值

| 模型 | Correctness | Faithfulness | Answer Relevancy | Contextual Relevancy |
| --- | ---: | ---: | ---: | ---: |
| GPT-4o-mini | 0.649 | 0.874 | **0.748** | **0.369** |
| Veri | **0.749** | **0.891** | 0.732 | 0.365 |
| DeepSeek | 0.650 | 0.863 | 0.738 | 0.363 |

### 4.2 指标达标率

| 模型 | Correctness | Faithfulness | Answer Relevancy | Contextual Relevancy |
| --- | ---: | ---: | ---: | ---: |
| GPT-4o-mini | 35.75% | 78.76% | **56.48%** | 17.62% |
| Veri | **72.28%** | **80.43%** | 52.72% | **18.48%** |
| DeepSeek | 39.18% | 77.32% | 54.12% | 18.04% |

Veri 的 Correctness 优势较大，说明其实际作答更接近参考答案。三者 Faithfulness 接近；Answer Relevancy 由 GPT 略高。Contextual Relevancy 只作诊断，不参与用例门控。

## 五、共同样本端到端结果

| 模型 | 正确回答数 / 有答案题 | Correct Answer Rate | 门控通过数 / 全部题 | Case Pass Rate |
| --- | ---: | ---: | ---: | ---: |
| GPT-4o-mini | 69 / 194 | 35.57% | 245 / 419 | **58.47%** |
| Veri | **133 / 194** | **68.56%** | 54 / 419 | 12.89% |
| DeepSeek | 76 / 194 | 39.18% | 237 / 419 | 56.56% |

Correct Answer Rate 更适合回答“应当回答的问题中，有多少最终给出了正确答案”。Case Pass Rate 还要求所有适用门控指标达标，因此更严格，也更容易受到拒答文本 Correctness 和回答风格影响。

## 六、限制与风险

1. **共同样本不是随机样本。** 419 题来自评估顺序形成的共同成功交集，可能存在顺序偏差。
2. **裁判不是独立的。** 三组结果均使用 GPT-4o-mini 裁判；GPT 考生存在自评偏差风险。
3. **这不是 DeepSeek 裁判报告。** 新配置下应生成 `*-judge-deepseek` 结果，再与本报告对照裁判稳定性。
4. **低 Contextual Relevancy 仅作诊断。** 当前每题使用整段保存上下文而非独立检索结果，该指标不参与 `case_passed`。

## 七、共同样本判断

- **若优先答案正确性：** 当前 Veri 最强，Correct Answer Rate 和 AA Correctness 均明显领先。
- **若优先回答/拒答边界：** 在这 419 题共同样本中，DeepSeek 最好；仍建议使用随机分层样本复核。
- **若优先简洁度和综合门控：** GPT 与 DeepSeek 接近，GPT 当前略高；二者 Correctness 均明显低于 Veri。
- **最终选型前：** 应完成 DeepSeek 全量，并用新配置下的 DeepSeek 裁判重新评估三个模型；同一模型既当考生又当裁判的结果应与人工抽样或独立裁判交叉验证。

## 八、来源目录

- `evaluation_results/20260914-111353-gpt-4o-mini-judge-gpt-4o-mini`
- `evaluation_results/20260915-004747-veri-judge-gpt-4o-mini`
- `evaluation_results/20260915-114244-deepseek-judge-gpt-4o-mini`

报告中的所有统计均固定在上述数据截点，并只使用三个模型共同成功评估的 419 道题。
