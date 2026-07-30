

- [用法](#用法)
- [Evaluation](#evaluation)
- [RAG问答系统评估指标](#rag问答系统评估指标)
  - [1. 回答决策的四种状态](#1-回答决策的四种状态)
  - [2. 回答决策指标](#2-回答决策指标)
  - [3. 回答质量指标](#3-回答质量指标)
  - [4. 检索与回答的完整评估维度](#4-检索与回答的完整评估维度)
  - [5. 各指标的区别](#5-各指标的区别)
  - [6. Superman案例](#6-superman案例)
  - [7. 建议采用的最终结构](#7-建议采用的最终结构)
    - [Retrieval Quality](#retrieval-quality)
    - [Generation Quality](#generation-quality)
    - [Statistical Outcomes](#statistical-outcomes)


# 用法

.env文件中写入 `OPENAI_API_KEY=xxxxxxxx`

然后执行命令

```bash
conda activate py312
pip install deepeval openai pyyaml
deepeval test run test_evaluation.py
```








更多metrics可以参考: `https://www.zdoc.app/zh/confident-ai/deepeval`

| Metrics / Capabilities | Chinese Description | DeepEval | RAGAS | RAGChecker | RefChecker | AbstentionBench |
|---|---|:---:|:---:|:---:|:---:|:---:|
| Faithfulness | 回答是否完全受检索内容支持 | ✅ | ✅ | ✅ | ✅ | ❌ |
| Answer Relevancy | 回答是否与问题相关、有没有答非所问 | ✅ | ✅ | ✅ | ❌ | ❌ |
| Contextual Precision | 检索到的内容中，有多少是真正相关的 | ✅ | ✅ | ✅ | ❌ | ❌ |
| Contextual Recall | 应该找到的关键证据是否都被检索出来 | ✅ | ✅ | ✅ | ❌ | ❌ |
| Response Correctness | 回答与标准答案是否一致 | ✅ | ✅ | ✅ | ❌ | ❌ |
| Claim-level Verification | 将回答拆成声明，逐条检查支持、矛盾或无依据 | ⚠️ Custom | ⚠️ Partial | ✅ | ✅ | ❌ |
| Hallucination Detection | 检查回答是否包含文档中没有的信息 | ✅ | ✅ | ✅ | ✅ | ⚠️ Partial |
| No Data Found / Abstention | 信息不足时，是否正确拒绝回答 | ⚠️ Custom | ⚠️ Custom | ⚠️ Custom | ⚠️ Custom | ✅ |
| Over-refusal Detection | 文档有答案时，系统是否错误拒答 | ⚠️ Custom | ⚠️ Custom | ⚠️ Custom | ❌ | ✅ |
| Citation Presence | 检查回答是否包含引用 | ⚠️ Custom | ⚠️ Custom | ⚠️ Custom | ❌ | ❌ |
| Citation Correctness | 检查引用是否真正支持回答 | ✅ | ✅ | ✅ | ✅ | ❌ |
| Citation Completeness | 回答中的关键声明是否都有引用支持 | ⚠️ Custom | ✅ | ✅ | ✅ | ❌ |
| Retrieval Diagnosis | 区分错误来自检索器还是回答生成器 | ⚠️ Partial | ⚠️ Partial | ✅ | ❌ | ❌ |
| SOP Step Completeness | SOP 必要步骤是否完整返回 | ⚠️ Custom | ⚠️ Custom | ⚠️ Custom | ⚠️ Custom | ❌ |
| SOP Step Order | SOP 步骤顺序是否正确 | ⚠️ Custom | ⚠️ Custom | ⚠️ Custom | ❌ | ❌ |
| Invented SOP Steps | 是否加入原始 SOP 中不存在的步骤 | ⚠️ Custom | ⚠️ Custom | ✅ | ✅ | ❌ |
| Dataset Generation | 根据文档自动生成测试问题和标准答案 | ✅ | ✅ | ⚠️ Limited | ❌ | ❌ |
| Regression Testing | 更换模型、Prompt 或知识库后重新运行并比较结果 | ✅ | ⚠️ External | ⚠️ External | ⚠️ External | ❌ |
| Pytest Integration | 可作为自动化测试在 CI/CD 中运行 | ✅ | ⚠️ Custom | ⚠️ Custom | ⚠️ Custom | ❌ |
| LLM-as-a-Judge | 使用另一个 LLM 自动评估回答质量 | ✅ | ✅ | ✅ | ✅ | ✅ |
| Best Use Case | 最适合的使用场景 | 自动化 API 测试与回归测试 | RAG 综合评分与测试集生成 | 深度诊断检索和生成错误 | 逐条声明与幻觉检查 | 专门测试模型是否知道自己不知道 |


符号含义：

- ✅：原生支持或主要功能
- ⚠️ Partial：部分支持，但不是核心能力
- ⚠️ Custom：可以实现，但需要编写自定义 Metric
- ⚠️ External：需要配合外部测试脚本或 CI 工具
- ❌：通常不支持或不适合该用途


# Evaluation

我们故意造出一些没有答案的题目，也就是说题目分成两类：
```
1. 有答案的题目
1.1 答对
1.2 答错
1.3 误判为无答案

2. 无答案的题目
2.1 拒答，知道无答案 
2.2 无中生有，给出的答案
```

wiki数据1000

- 判断材料是否足以回答问题（Answerability / Abstention）:模型是否正确判断材料足不足以回答问题。
- 回答正确性（Correctness）:
- 相关性（Answer Relevancy）
- 是否忠于材料（Faithfulness）

用chatGPT模型=》篡改+生成问题

然后去遍历：VeriAI、chatGPT系列模型、Gemini模型

我的整理：

ChatGPT设计问题：
```
有答案
  答有（可从材料找到出处，会明确回答“有相关”）
  答无（不可找到，会明确回答“无相关”）
无答案
  拒答（不可找到，会明确回答“无相关”）
```

其中认为“答无”和“拒答”类似：

| 数据类型 | 模型行为 | 分类  |
| ---- | ---- | --- |
| 有答案  | 正确回答 | 1.1 |
| 有答案  | 错误回答 | 1.2 |
| 有答案  | 拒绝回答 | 1.3 |
| 无答案  | 拒绝回答 | 2.1 |
| 无答案  | 给出答案 | 2.2 |




```
比如对2026年世界杯的内容，故意问2025和2027年。
- 其中问2025年，明确回答“无”，但是给了2026年的相近推荐。
- 问2027年，明确回答“无此资料”。

問題：2025年世界盃決賽的比分是多少？
我幫您看了一下資料，目前知識庫中並沒有關於「2025年世界盃」的相關資訊喔。 不過，我找到了 **2026年國際足協世界盃** 的決賽結果：在2026年7月19日的比賽中，西班牙以 **1–0** 擊敗了阿根廷，奪得冠軍。 如果您是想查詢其他年份的賽事，或是有關於2026年世界盃的其他疑問，歡迎隨時告訴我，我很樂意為您解答！😊
🔊
引用段落 (1)：

2026年國際足協世界盃決賽是2026年世界盃足球賽最後一場賽事，於2026年7月19日在美國新澤西州東盧瑟福的大都會人壽運動場（賽事期間稱為「紐約新澤西運動場」）舉行，由西班牙以1–0擊敗衛冕冠軍阿根廷，贏得世界盃冠軍，也是首次在北美洲奪冠的歐洲球隊。
問題：2027年世界盃決賽的比分是多少？
【未搜索到知识库中的相关信息】
🔊
引用段落 (0)：
```




# RAG问答系统评估指标

## 1. 回答决策的四种状态

“有答案/无答案”表示材料是否包含问题所需的信息；“回答/拒答”表示模型是否给出了实质答案。

| 缩写 | 材料实际情况 | 模型行为 | 中文描述 | 评价                 |
| -- | ------ | ---- | ---- | ------------------ |
| AA | 有答案    | 给出回答 | 有答有  | 回答决策正确，但答案内容不一定正确  |
| NN | 无答案    | 明确拒答 | 无答无  | 正确拒答               |
| AN | 有答案    | 明确拒答 | 有答无  | 错误拒答，模型过于保守        |
| NA | 无答案    | 给出回答 | 无答有  | 强行作答，可能属于无中生有或答非所问 |

> 明确拒答是指模型表示“无法从材料或知识库中找到相关信息”，而不仅仅是没有直接回答问题。

## 2. 回答决策指标

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

## 3. 回答质量指标

“有答有”只表示材料有答案，而且模型选择了回答，并不代表模型一定答对。因此，还需要评估回答本身的质量。

| 指标               | 比较对象                                | 核心问题            | 主要用途              |
| ---------------- | ----------------------------------- | --------------- | ----------------- |
| Correctness      | `actual_output ↔ expected_output`   | 模型回答是否正确？       | 判断答案是否符合标准答案      |
| Faithfulness     | `actual_output ↔ retrieval_context` | 模型回答是否受到材料支持？   | 识别与材料矛盾或没有依据的内容   |
| Answer Relevancy | `actual_output ↔ input`             | 模型是否真正回答了用户的问题？ | 识别答非所问或只提供相邻信息的回答 |

## 4. 检索与回答的完整评估维度

| 评估阶段 | 评估维度                                   | 比较对象                                  | 评估问题               |
| ---- | -------------------------------------- | ------------------------------------- | ------------------ |
| 检索阶段 | Contextual Relevancy                   | `input ↔ retrieval_context`           | 检索到的材料是否与问题相关？     |
| 材料判断 | Context Answerability                  | `input + retrieval_context`           | 当前材料是否包含回答问题所需的信息？ |
| 回答生成 | Answer Relevancy                       | `input ↔ actual_output`               | 模型是否真正回答了问题？       |
| 回答生成 | Correctness                            | `expected_output ↔ actual_output`     | 模型回答是否正确？          |
| 回答生成 | Faithfulness                           | `retrieval_context ↔ actual_output`   | 模型回答是否完全得到材料支持？    |

## 5. 各指标的区别

| 情况               | Contextual Relevancy | Context Answerability | Answer Relevancy | Correctness | Faithfulness |
| ---------------- | -------------------- | --------------------- | ---------------- | ----------- | ------------ |
| 检索到完全无关的材料       | 低                    | 通常为无答案                | 取决于回答            | 取决于回答       | 取决于回答        |
| 材料主题相关，但缺少问题所需信息 | 中或高                  | 无答案                   | 取决于回答            | 取决于回答       | 取决于回答        |
| 回答正确且有材料依据       | 高                    | 有答案                   | 高                | 高           | 高            |
| 回答符合材料，但没有回答问题   | 可能高                  | 可能有或无答案               | 低                | 低           | 高            |
| 回答切中问题，但内容是编造的   | 可能低                  | 无答案                   | 高                | 低           | 低            |
| 材料无答案，模型明确拒答     | 可能低或中                | 无答案                   | 高                | 高           | 不适用或高        |

## 6. Superman案例

```text
Question:
When was Superman born?

Retrieved context:
Superman was born Kal-El, on the fictional planet Krypton.

Actual output:
Superman was born on Krypton, and his original name was Kal-El.
```

| 评估维度                                   | 结果  | 原因                       |
| -------------------------------------- | --- | ------------------------ |
| Contextual Relevancy                   | 中等  | 材料与Superman出生相关，但没有时间信息  |
| Context Answerability                  | 无答案 | 材料没有出生日期、年份或时间           |
| Answerability / Abstention Correctness | 低   | 模型没有说明材料缺少出生时间           |
| Answer Relevancy                       | 低   | 问题问“When”，回答却提供了地点和姓名    |
| Correctness                            | 低   | 没有给出预期的“材料未提供出生时间”       |
| Faithfulness                           | 高   | Krypton和Kal-El都来自材料，没有编造 |

这个案例说明：Faithfulness高并不意味着回答正确，也不意味着回答切中了问题。

## 7. 建议采用的最终结构

### Retrieval Quality

| 维度                    | 作用              |
| --------------------- | --------------- |
| Contextual Relevancy  | 判断检索到的材料是否与问题相关 |
| Context Answerability | 判断材料是否足以回答问题    |

### Generation Quality

| 维度                                     | 作用              |
| -------------------------------------- | --------------- |
| Answerability / Abstention Correctness | 判断模型是否正确选择回答或拒答 |
| Answer Relevancy                       | 判断模型是否真正回答了问题   |
| Correctness                            | 判断答案是否正确        |
| Faithfulness                           | 判断答案是否忠于材料      |

### Statistical Outcomes

| 统计结果 | 含义           |
| ---- | ------------ |
| AA   | 材料有答案，模型给出回答 |
| NN   | 材料无答案，模型正确拒答 |
| AN   | 材料有答案，模型错误拒答 |
| NA   | 材料无答案，模型仍然作答 |

最终可以通过AA、NN、AN和NA计算决策准确率、错误拒答率、无中生有率、回答精确率、回答召回率和拒答精确率。
