# 项目代理指南

本仓库用于构建 Wikipedia 问答测试集、让被测模型作答，并用 DeepEval 评估 RAG/问答系统。

## 开始工作前

1. 先读 `docs/STATUS.md`，了解当前数据规模、已完成工作和未解决问题。
2. 涉及流程、字段或指标时，再读 `docs/ARCHITECTURE.md`。
3. `readme.md` 面向使用者；架构与状态以 `docs/` 为维护入口。
4. 不要为了理解项目遍历 `evaluation_cases/` 或 `evaluation_results/`；它们很大，优先读取配置、入口脚本和状态文档。

## 必须保持的契约

- `config.yaml` 的 `target.model` 是被测模型，控制作答字段后缀；`judge.model` 是 DeepEval 裁判模型，两者不得混用。
- 题目生成时，无后缀的 `actual_answered` 和 `actual_output` 是 `null` 占位。
- 作答结果优先写入并读取 `actual_answered_<target.model>` 与 `actual_output_<target.model>`；无后缀字段只用于旧数据兼容。
- 作答器和评估器必须使用 JSON 用例中保存的 `retrieval_context`。不要用提取出的完整 TXT 替换它，否则可能泄漏出题时截断范围之外的信息。
- 回答决策只有 AA、NN、AN、NA 四种状态。Correctness 始终参与用例判定；Answer Relevancy 和 Faithfulness 仅在实际作答时参与；Contextual Relevancy 仅作诊断。
- 长任务必须保持逐题原子保存和断点恢复能力；不要破坏现有输出或使用破坏性 Git 命令。
- `evaluation_results/` 中的交互日志可能包含完整上下文和模型回答，应按敏感数据处理。

## 验证与成本边界

先运行无网络测试：

```bash
source .venv/bin/activate
python -m pytest -q test_evaluation_logic.py
python -m py_compile evaluation.py gpt-4o-mini_answer.py generate_wikipedia_test_cases.py
```

以下命令会访问 API、产生费用或耗时，除非用户明确要求，否则不要自行运行：

```bash
python generate_wikipedia_test_cases.py
python gpt-4o-mini_answer.py
python evaluation.py
python -m pytest -q test_chatbot.py test_veris.py
```

不要提交 `.env`、API key、Bearer token 或完整交互日志。修改核心契约后，同步更新 `docs/ARCHITECTURE.md`；完成运行、发现风险或改变优先级后，同步更新 `docs/STATUS.md`。

## Git 提交约定

用户明确要求完成 Git 提交时，直接执行：

```bash
git add .
git commit -m "<简短说明>"
git push
```
