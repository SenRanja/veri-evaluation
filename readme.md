

- [用法](#用法)
- [问答系统评估指标](#问答系统评估指标)
  - [回答决策的四种状态](#回答决策的四种状态)
  - [回答决策指标](#回答决策指标)
  - [回答质量指标](#回答质量指标)
  - [指标适用性与用例判定](#指标适用性与用例判定)
  - [汇总统计](#汇总统计)
  - [评估指标说明](#评估指标说明)
    - [举例：门店的退款政策](#举例门店的退款政策)
- [测试数据](#测试数据)
  - [构建思路](#构建思路)
  - [生成测试数据](#生成测试数据)


# 用法

.env文件中写入 `OPENAI_API_KEY=xxxxxxxx`

然后执行命令

```bash
conda activate py312
pip install -r requirements.txt
python evaluation.py
```

`config.yaml` 中 `target.model` 表示被测模型，决定作答字段后缀；`judge.model` 表示 DeepEval 裁判模型。两者可以不同。



# 问答系统评估指标

## 回答决策的四种状态


```
对于布尔字段 `actual_answered`：

以下情况均记为 `actual_answered = false`：

- 模型明确拒绝回答；
- 模型表示材料中没有相关信息；
- 模型虽然输出了内容，但没有针对用户实际询问的内容给出实质答案。

只要模型针对用户实际询问的内容给出了明确的实质答案，就记为 `actual_answered = true`，无论该答案是否正确、是否完整或是否得到材料支持。

`expected_answered` 和 `actual_answered` 用于判断AA、NN、AN和NA。需要注意，AA只表示模型选择了回答，不表示回答内容正确；回答质量需要由Correctness、Faithfulness和Answer Relevancy进一步评价。
```

故而设置 bool 类型的 `expected_answered` 和 `actual_answered`。

因此测试用例如：

```json
[
  {
    "name": "refund_policy_document",
    "retrieval_context": [
      "All customers are eligible for a 30-day full refund at no extra costs."
    ],
    "questions": [
      {
        "name": "correct_refund_policy",
        "input": "What if these shoes don't fit?",
        "expected_answered": true,
        "actual_answered_gpt-4o-mini": true,
        "actual_output_gpt-4o-mini": "You have 30 days to return them for a full refund at no extra cost.",
        "expected_output": "We offer a 30-day full refund at no extra costs."
      }
    ]
  }
]
```

| 状态 | expected_answered | actual_answered | 含义           | 决策结果          |
| -- | ----------------: | --------------: | ------------ | ------------- |
| AA |            `true` |          `true` | 材料有答案，模型给出回答 | 正确决策，但答案不一定正确 |
| NN |           `false` |         `false` | 材料无答案，模型没有回答 | 正确决策          |
| AN |            `true` |         `false` | 材料有答案，模型没有回答 | 错误拒答          |
| NA |           `false` |          `true` | 材料无答案，模型仍然回答 | 不受材料支持的作答     |

字段命名约定：题目生成阶段 `actual_answered` 与 `actual_output` 为 `null` 占位；被测系统作答后，结果根据 `target.model` 写入带模型后缀的字段，例如 `actual_answered_gpt-4o-mini` 与 `actual_output_gpt-4o-mini`。评估器读取同一后缀字段，因此同一份题目集可以同时保存多个被测模型的作答结果。为兼容旧数据，后缀字段不存在时评估器会回退读取无后缀字段。

## 回答决策指标

| 指标                       | 计算公式                | 说明                     | 趋势   |
| ------------------------ | ------------------- | ---------------------- | ---- |
| Decision Accuracy        | `(AA + NN) / Total` | 模型正确决定回答或拒答的比例         | 越高越好 |
| False Refusal Rate       | `AN / (AA + AN)`    | 材料有答案，但模型错误拒答的比例       | 越低越好 |
| Hallucinated Answer Rate | `NA / (NN + NA)`    | 材料无答案，但模型仍然给出答案的比例     | 越低越好 |
| Answer Precision         | `AA / (AA + NA)`    | 模型选择回答时，材料实际上包含答案的比例   | 越高越好 |
| Answer Recall            | `AA / (AA + AN)`    | 材料包含答案时，模型愿意回答的比例      | 越高越好 |
| Abstention Precision     | `NN / (NN + AN)`    | 模型选择拒答时，材料实际上确实没有答案的比例 | 越高越好 |

其中：

```text
Answer Recall = 1 - False Refusal Rate
```

报告比率时必须同时给出各状态的计数。分母为 0 的比率记为 `null`，不得记为 0 或 1。题目数量较少时单个比率波动较大，跨运行比较应结合计数解读。

## 回答质量指标

“有答有”只表示材料有答案，而且模型选择了回答，并不代表模型一定答对。因此，还需要评估回答本身的质量。

| 指标               | 比较对象                                | 核心问题            | 主要用途              |
| ---------------- | ----------------------------------- | --------------- | ----------------- |
| Correctness      | `actual_output ↔ expected_output`   | 模型回答是否正确？       | 判断答案是否符合标准答案      |
| Faithfulness     | `actual_output ↔ retrieval_context` | 模型回答是否受到材料支持？   | 识别与材料矛盾或没有依据的内容   |
| Answer Relevancy | `actual_output ↔ input`             | 模型是否真正回答了用户的问题？ | 识别答非所问或只提供相邻信息的回答 |

三个指标各自独立打分，互不推导，组合起来才有诊断价值：

| Correctness | Faithfulness | 含义 |
| --- | --- | --- |
| 高 | 高 | 正常好回答 |
| 高 | 低 | 碰巧答对，答案并非来自材料（依赖模型自身知识） |
| 低 | 高 | 引用了材料但没有答对问题要点 |
| 低 | 低 | 无依据编造（幻觉） |

## 指标适用性与用例判定

四个指标并非对所有用例都有意义：

- 拒答文本不包含事实主张，Faithfulness 无从校验；
- 判题模型经常把拒答判为“没有回答问题”，Answer Relevancy 对拒答的打分不可靠；
- 无答案用例的材料与问题“相关性低”本身就是构造目标，Contextual Relevancy 必然偏低。

如果要求每个用例通过全部四个指标，正确拒答（NN）几乎必然被误判为失败。因此指标是否参与用例判定（gate）由 `actual_answered` 决定——门控条件是“是否实际作答”，而不是“是否答对”：

| 指标 | actual_answered = true | actual_answered = false | 原因 |
| --- | --- | --- | --- |
| Correctness | 参与判定 | 参与判定 | 拒答时同样要求拒答表述与预期一致 |
| Faithfulness | 参与判定 | 仅诊断 | 拒答没有可校验的事实主张 |
| Answer Relevancy | 参与判定 | 仅诊断 | 对拒答的打分不可靠 |
| Contextual Relevancy | 仅诊断 | 仅诊断 | 本流程无真实检索器，材料即出题来源 |

判定规则：

```text
decision_passed = 决策状态 ∈ {AA, NN}
case_passed    = decision_passed 且 所有“参与判定”的指标分数 ≥ 阈值
```

“仅诊断”的指标照常计算并写入结果，但不影响 `case_passed`，也不计入下述条件均值。

`results.json` 中每个指标使用 `gates_case` 标明是否参与当前用例判定；`summary.csv` 使用对应的 `metric_gates_case` 列。`passed` 作为旧字段保留，其值与规范字段 `case_passed` 相同。

指标还包含稳定的机器字段 `id`（`correctness`、`faithfulness`、`answer_relevancy`、`contextual_relevancy`）。判定与汇总按 `id` 工作，`name` 只用于显示，避免 DeepEval 版本差异改变逻辑。

## 汇总统计

质量分数必须按决策状态条件化统计，避免把“拒答质量”与“作答质量”混为一谈：

| 统计项 | 统计范围 | 含义 |
| --- | --- | --- |
| 平均 Correctness、Faithfulness、Answer Relevancy | 仅 AA 用例 | 模型作答且材料有答案时的回答质量 |
| 平均 Correctness | 仅 NN 用例 | 拒答表述是否符合预期拒答 |
| Correct Answer Rate | `#(AA 且 Correctness ≥ 阈值) / #(expected_answered = true)` | 端到端正确回答率：既敢答又答对 |

Correct Answer Rate 是最贴近用户体验的单一指标：错误拒答（AN）与答错都会拉低它。

Correctness、Faithfulness 与 Answer Relevancy 含义不同，不要把三者直接平均成一个总分。需要最终数据时按此取用：

- 只汇报一个数：Correct Answer Rate；
- 汇报两个数：再加 Decision Accuracy；
- 逐用例结论：`case_passed`（决策正确且所有参与判定的指标 ≥ 阈值），其通过率可作总览。

所有统计项输出时附带分子与分母；分母为 0 记为 `null`。

## 参考答案审计

GPT、Gemini、Veri 中至少两个模型有完整历史结果，且可用模型出现决策不一致或一致得到 NA/AN 时，可以调用审核脚本检查 golden label。Gemini 没有作答的题目仍会使用 GPT 与 Veri 结果进入候选：

```bash
source .venv/bin/activate
python -u revise_reference_answers.py --model gpt-4o-mini --limit 10
python -u revise_reference_answers.py --model gpt-4o-mini
python -u revise_reference_answers.py --model gpt-4o-mini --apply
bash evaluation.sh
```

前两条命令只更新 `evaluation_results/reference_answer_revision_audit.json`，不会修改用例。`--apply` 只应用审核模型标为高置信、无歧义且无需人工复核的建议，并原子更新 `test_cases_novel.json`。脚本不上传文件，只把用例中的精确 `retrieval_context`、当前参考答案和 GPT/Gemini/Veri 的可用历史回答作为文本交给审核模型，golden 字段严格以 `retrieval_context` 为准。

修订用例不会改变历史 `results.json`。应用后必须重新运行 `evaluation.sh`，才能得到基于新参考答案的三个目标模型评估结果。

评估运行开始后会立即创建 `results.json`。每收到一个指标响应，程序都会通过临时文件原子替换该 JSON，因此 Ctrl+C 或异常退出时，已经返回的指标不会丢失。文件中的 `progress` 包含：

- `status`：`running` 或 `completed`；
- `total_cases` / `completed_cases`；
- `metric_responses` / `expected_metric_responses`。

尚未完成的用例以 `status: in_progress` 保存，`case_passed` 为 `null`；实时 Decision/Quality 汇总只统计 `status: completed` 的用例，防止部分指标污染总体数据。每完成一个用例，终端会即时输出完成数、通过数、Decision Accuracy 和 Correct Answer Rate。

## 评估指标说明

四个指标分别评价RAG系统中的不同关系：

```
问题 --(Contextual Relevancy)-- 材料
材料 --(Faithfulness)-- 回答
问题 --(Answer Relevancy)-- 回答
标准答案 --(Correctness)-- 模型回答
```

| 指标                   | 比较对象        | 作用                 |
| -------------------- | ----------- | ------------------ |
| Contextual Relevancy | 问题 ↔ 检索材料   | 判断系统是否检索到了与问题相关的材料 |
| Answer Relevancy     | 问题 ↔ 模型回答   | 判断模型是否直接回答了用户的问题   |
| Correctness          | 标准答案 ↔ 模型回答 | 判断模型回答是否正确         |
| Faithfulness         | 检索材料 ↔ 模型回答 | 判断模型回答是否得到检索材料支持。   |

所有metrics分值在[0, 1].

1. Contextual Relevancy:
判断检索材料与用户问题是否相关，主要用于评价检索阶段。

需要注意，材料“与问题相关”不代表材料“一定足以回答问题”。例如，材料可能讨论退款政策，但没有提供用户询问的退款门店地址。

Contextual Relevancy主要作为诊断指标。对于故意构造的无答案测试用例，该指标必然偏低；且本流程没有真实检索器，材料即出题来源。因此该指标在任何用例中都不参与 `case_passed` 判定，回答或拒答决策根据AA、NN、AN和NA独立评价。

2. Answer Relevancy:
判断模型是否直接回答了用户的问题。

该指标不判断回答是否正确，也不判断回答是否有材料依据。
实际判题中，明确说明“当前材料没有相关信息”的拒答经常被打为低分甚至 0 分，因此该指标只在 `actual_answered = true` 时参与判定；拒答用例中仅作诊断记录。

3. Correctness:

判断模型回答是否正确。
看模型的回答（包括回答，或者回答无，或者拒绝回答）是否与预期回答的内容一致。

4. Faithfulness:

判断模型回答中的事实陈述是否得到检索材料支持，以及回答是否与材料存在事实矛盾。该指标主要用于识别模型基于材料进行回答时产生的编造或曲解。拒答不包含事实主张，因此该指标只在 `actual_answered = true` 时参与判定。


### 举例：门店的退款政策

例如：

```text
问题：
应该去哪一家悉尼门店退款？

检索材料：
所有顾客均可在30天内获得全额退款。

模型回答：
应该前往Sydney CBD门店。

标准答案：
提供的材料没有说明应该前往哪一家悉尼门店。
```


可能得到以下结果：

```text
Contextual Relevancy：低或中等
Answer Relevancy：高
Correctness：低
Faithfulness：低
```

按上述判定规则：`expected_answered=false` 而模型作答，决策状态为 NA，`decision_passed=false`，用例直接不通过；Correctness 与 Faithfulness 的低分进一步说明该回答属于无依据编造。Contextual Relevancy 只做诊断，不影响判定。


# 测试数据

## 构建思路

我自己下载了wiki的一些东西（`./wiki_downloader/wiki_downloader.py`）下载到文件 `./evaluation_cases/wikipedia_10000.jsonl` 中，以此来构建测试数据。

单行如：

```json
{"page_id": 70533387, "title": "Ahmad Bazzi", "text": "Ahmad Bazzi is a French-Lebanese research scientist at NYU WIRELESS, New York University Tandon School of Engineering and New York University Abu Dhabi. He is an inventor of different patents in the field of wireless communications, and more specifically in Bluetooth technologies. The patent is in market as it doubles the range of Bluetooth Low Energy devices and reduce power consumption by 90 percent.\nIn addition to his research, Bazzi is an educator on YouTube, where he publishes engineering and programming topics for a global audience.\n\n\n== References ==", "url": "https://en.wikipedia.org/wiki/Ahmad_Bazzi"}
```

```
python analyze_jsonl_characters.py 
文件：evaluation_cases\wikipedia_10000.jsonl
总行数：4970
平均数：3554.61
最长：142136 characters（行号：4352）
最短：306 characters（行号：1057）
中位数：1781
众数：355（各出现 12 次）
```


这是一个目录： `./evaluation_cases/test_cases_novel`

这是一个json文件： `./evaluation_cases/test_cases_novel.json`，如下：

```json
[
  {
    "name": "refund_policy_document",
    "retrieval_context": [
      "All customers are eligible for a 30-day full refund at no extra costs."
    ],
    "questions": [
      {
        "name": "correct_refund_policy",
        "input": "What if these shoes don't fit?",
        "expected_answered": true,
        "actual_answered_gpt-4o-mini": true,
        "actual_output_gpt-4o-mini": "You have 30 days to return them for a full refund at no extra cost.",
        "expected_output": "We offer a 30-day full refund at no extra costs."
      },
      {
        "name": "wrong_refund_period",
        "input": "How many days do I have to return the shoes?",
        "expected_answered": true,
        "actual_answered_gpt-4o-mini": true,
        "actual_output_gpt-4o-mini": "You can return them within 90 days.",
        "expected_output": "You can return them within 30 days."
      },
      {
        "name": "refund_store_location",
        "input": "Which Sydney store should I visit for the refund?",
        "expected_answered": false,
        "actual_answered_gpt-4o-mini": false,
        "actual_output_gpt-4o-mini": "The provided material does not specify a store location.",
        "expected_output": "The provided material does not specify a store location."
      },
      {
        "name": "refund_store_hallucination",
        "input": "Which Sydney store should I visit for the refund?",
        "expected_answered": false,
        "actual_answered_gpt-4o-mini": true,
        "actual_output_gpt-4o-mini": "You should visit the Sydney CBD store.",
        "expected_output": "The provided material does not specify a store location."
      }
    ]
  }
]
```

我希望从`./evaluation_cases/wikipedia_10000.jsonl`中提取`text`字段内容写成文件在`{page_id}-{title}.txt`的文件中；另外我需要大模型围绕我的方案，设计问题，考察


生成时 `actual_answered` 和 `actual_output` 默认为 `null`；被测系统作答后写入 `actual_answered_<model>` 与 `actual_output_<model>` 字段（见上文字段命名约定）

## 生成测试数据

将jsonl文件提取text到文件中： `python extract_wikipedia_texts.py`

由文件生成到json的问题集： `python generate_wikipedia_test_cases.py --limit 100`

`generate_wikipedia_test_cases.py` 有断点恢复的能力，会按照jsonl文件从前往后进行问题集生成。比如从1生成到 正在处理50，我按了ctrl，下次我重新运行此命令，会从50开始重新生成并向后生成。

linux上运行:

```bash
# install pip requirements
source .env
python3 -u generate_wikipedia_test_cases.py \
  --limit 2000 \
  > log_gen_test_case.log 2>&1
```

生成完成后：

1. 让被测系统作答：`python gpt-4o-mini_answer.py`，严格使用每篇题目保存的 `retrieval_context`，并将结果写入 `actual_answered_<target.model>` 与 `actual_output_<target.model>`；
2. 运行评估：`python evaluation.py`，按[指标适用性与用例判定](#指标适用性与用例判定)规则打分，并输出[汇总统计](#汇总统计)。

提取出的 TXT 文件只用于上传到外部 RAG 系统或人工检查；内置作答脚本不读取 TXT，以免完整文章与生成题目时截断保存的 `retrieval_context` 不一致。




