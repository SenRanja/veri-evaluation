
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






