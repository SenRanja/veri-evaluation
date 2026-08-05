

- [用法](#用法)
- [问答系统评估指标](#问答系统评估指标)
  - [回答决策的四种状态](#回答决策的四种状态)
  - [回答决策指标](#回答决策指标)
  - [回答质量指标](#回答质量指标)
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
pip install deepeval openai pyyaml pydantic 
deepeval test run test_evaluation.py
```



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
        "actual_answered": true,
        "actual_output": "You have 30 days to return them for a full refund at no extra cost.",
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

## 回答质量指标

“有答有”只表示材料有答案，而且模型选择了回答，并不代表模型一定答对。因此，还需要评估回答本身的质量。

| 指标               | 比较对象                                | 核心问题            | 主要用途              |
| ---------------- | ----------------------------------- | --------------- | ----------------- |
| Correctness      | `actual_output ↔ expected_output`   | 模型回答是否正确？       | 判断答案是否符合标准答案      |
| Faithfulness     | `actual_output ↔ retrieval_context` | 模型回答是否受到材料支持？   | 识别与材料矛盾或没有依据的内容   |
| Answer Relevancy | `actual_output ↔ input`             | 模型是否真正回答了用户的问题？ | 识别答非所问或只提供相邻信息的回答 |


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

Contextual Relevancy主要作为检索质量诊断指标。对于故意构造的无答案测试用例，该指标可能较低，因此不应单独据此判断模型的拒答行为是否正确。回答或拒答决策应根据AA、NN、AN和NA独立评价。

2. Answer Relevancy:
判断模型是否直接回答了用户的问题。

该指标不判断回答是否正确，也不判断回答是否有材料依据。
明确说明“当前材料没有相关信息”的拒答通常仍然具有较高相关性，因为它直接回应了用户问题；但具体分数由评判模型决定，并不保证固定为1。

3. Correctness:

判断模型回答是否正确。
看模型的回答（包括回答，或者回答无，或者拒绝回答）是否与预期回答的内容一致。

4. Faithfulness:

判断模型回答中的事实陈述是否得到检索材料支持，以及回答是否与材料存在事实矛盾。该指标主要用于识别模型基于材料进行回答时产生的编造或曲解。


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
        "actual_answered": true,
        "actual_output": "You have 30 days to return them for a full refund at no extra cost.",
        "expected_output": "We offer a 30-day full refund at no extra costs."
      },
      {
        "name": "wrong_refund_period",
        "input": "How many days do I have to return the shoes?",
        "expected_answered": true,
        "actual_answered": true,
        "actual_output": "You can return them within 90 days.",
        "expected_output": "You can return them within 30 days."
      },
      {
        "name": "refund_store_location",
        "input": "Which Sydney store should I visit for the refund?",
        "expected_answered": false,
        "actual_answered": false,
        "actual_output": "The provided material does not specify a store location.",
        "expected_output": "The provided material does not specify a store location."
      },
      {
        "name": "refund_store_hallucination",
        "input": "Which Sydney store should I visit for the refund?",
        "expected_answered": false,
        "actual_answered": true,
        "actual_output": "You should visit the Sydney CBD store.",
        "expected_output": "The provided material does not specify a store location."
      }
    ]
  }
]
```

我希望从`./evaluation_cases/wikipedia_10000.jsonl`中提取`text`字段内容写成文件在`{page_id}-{title}.txt`的文件中；另外我需要大模型围绕我的方案，设计问题，考察


`actual_answered` 和 `actual_output` 默认为 `null`

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




