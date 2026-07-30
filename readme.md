

- [用法](#用法)
- [问答系统评估指标](#问答系统评估指标)
  - [回答决策的四种状态](#回答决策的四种状态)
  - [回答决策指标](#回答决策指标)
  - [回答质量指标](#回答质量指标)
  - [评估指标说明](#评估指标说明)
    - [举例：门店的退款政策](#举例门店的退款政策)


# 用法

.env文件中写入 `OPENAI_API_KEY=xxxxxxxx`

然后执行命令

```bash
conda activate py312
pip install deepeval openai pyyaml
deepeval test run test_evaluation.py
```



# 问答系统评估指标

## 回答决策的四种状态

“有答案/无答案”表示材料是否包含问题所需的信息；“回答/拒答”表示模型是否给出了实质答案。

```
拒绝回答 = No
回答“材料中没有相关信息” = No
没有回答到问题 = No
给出实质答案 = Yes

不过这里无法解决：
1. 不区分“有答案并正确回答”和“有答案但回答错误”
2. 材料无答案并正确拒答 和 材料无答案但强行回答
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


| 缩写 | 材料实际情况 | 模型行为 | 中文描述 | 评价                 |
| -- | ------ | ---- | ---- | ------------------ |
| AA | 有答案    | 给出回答 | 有答有  | 回答决策正确，但答案内容不一定正确  |
| NN | 无答案    | 明确拒答 | 无答无  | 正确拒答               |
| AN | 有答案    | 明确拒答 | 有答无  | 错误拒答，模型过于保守        |
| NA | 无答案    | 给出回答 | 无答有  | 强行作答，可能属于无中生有或答非所问 |

| expected_answered | actual_answered | 状态 | 含义              |
| ----------------: | --------------: | -- | --------------- |
|            `true` |          `true` | AA | 应该回答，而且模型回答了    |
|           `false` |         `false` | NN | 应该不回答，而且模型没有回答  |
|            `true` |         `false` | AN | 应该回答，但模型拒答或没有回答 |
|           `false` |          `true` | NA | 不应该回答，但模型仍然给出答案 |


> 明确拒答是指模型表示“无法从材料或知识库中找到相关信息”，而不仅仅是没有直接回答问题。

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

| 指标                   | 比较对象        | 作用                 |
| -------------------- | ----------- | ------------------ |
| Contextual Relevancy | 问题 ↔ 检索材料   | 判断系统是否检索到了与问题相关的材料 |
| Answer Relevancy     | 问题 ↔ 模型回答   | 判断模型是否直接回答了用户的问题   |
| Correctness          | 标准答案 ↔ 模型回答 | 判断模型回答是否正确         |
| Faithfulness         | 检索材料 ↔ 模型回答 | 判断模型回答是否得到检索材料支持。   |

所有metrics分值在[0, 1].

1. Contextual Relevancy:
判断系统是否检索到了与问题相关的材料（这是评价检索阶段，在模型回答之前判断 问题 是否和 材料 有关）。
如果材料的确与问题无关，比如问题是问去哪里退款，但是材料没有说位置，那么就得分低；如果材料与问题有关，比如问题是退款额度，而且材料有退款额度的明确说明，那就得分高。
2. Answer Relevancy:
判断模型是否直接回答了用户的问题。
即便材料提到问题问的东西，或者没有提到，只要模型回答是切题问题的，就认为得分高。其中，据答和回答“没有”都默认得分1。
3. Correctness:
判断模型回答是否正确。
看模型的回答（包括回答，或者回答无，或者拒绝回答）是否与预期回答的内容一致。
4. Faithfulness:
判断模型回答是否得到检索材料支持（这是评价模型回答后阶段，在模型回答之后判断 模型回答 是否和 材料 有关）


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
Contextual Relevancy: 0.00
Answer Relevancy:     1.00
Correctness:          1.00
Faithfulness:         1.00
```
