# 项目状态

更新日期：2026-08-12

## 当前可用状态

- Python 依赖已写入根目录 `requirements.txt`。
- 本地 `.venv` 使用 Python 3.12.13，核心依赖可导入。
- `config.yaml` 当前设置：`target.model=gpt-4o-mini`、`judge.model=gpt-4o-mini`、`evaluation.max_workers=4`、四项阈值均为 0.7。
- Wikipedia 源数据 `evaluation_cases/wikipedia_10000.jsonl` 当前有 4,970 行。
- 当前用例 `evaluation_cases/test_cases_novel.json` 有 38 篇文档、152 道题。
- 152 道题均已有 `gpt-4o-mini` 模型后缀作答字段。
- 作答器已改为只使用用例 JSON 中保存的 `retrieval_context`，不再读取外部完整 TXT。
- 评估逻辑已实现目标/裁判模型分离、模型字段选择、指标适用性、条件质量汇总和 CSV 门控标记。
- `results.json` 现在从运行开始存在，并在每个 metric 响应后原子更新；总体汇总只统计完整用例，每完成一题即时输出实时总体指标。
- 指标门控和汇总已使用稳定内部 ID，不再依赖 DeepEval 可为空的 `name`。
- `test_evaluation_logic.py` 的 8 项离线测试已通过，覆盖每响应落盘、真实指标 ID、门控、字段选择和条件汇总。
- 当时也已通过核心 Python 文件编译和 VS Code 诊断检查。

## 尚未完成

- 新评估逻辑尚未完成一次全量 API 评估。最新目录 `evaluation_results/20260812-143856-gpt-4o-mini/` 只有 `config_snapshot.yaml` 和 `openai_interactions.jsonl`，属于中断/未完成运行，不能作为最终结果。
- 未确认新的完整运行能否稳定生成 `results.json`、`summary.csv` 和 `token_summary.json`。
- 未运行网络测试 `test_chatbot.py` 与 `test_veris.py`，它们会调用 LLM 并产生费用。

## 已知问题与风险

### 数据路径不一致

`wiki_downloader/wiki_downloader.py` 默认写入 `wiki_downloader/data/wikipedia_10000.jsonl`，而生成器和提取器默认读取 `evaluation_cases/wikipedia_10000.jsonl`。新下载流程需要手动移动/指定路径，或后续统一默认值。

### 成本和稳定性

- `generate_wikipedia_test_cases.py`、`gpt-4o-mini_answer.py`、`evaluation.py` 都会调用 API。
- 完整评估每题运行四项 LLM 指标；152 题会产生大量请求。
- 之前以 16 并发运行时留下未完成目录；当前已降为 4，但尚未完成全量验证。
- `test_chatbot.py` 和 `test_veris.py` 是网络测试，其中存在硬编码裁判模型，不应作为默认离线测试运行。

### 日志与凭据

- `openai_interactions.jsonl` 可包含完整问题、上下文和回答。
- `.env` 和任何 Bearer/API token 不得提交。
- `Veris_LLM_usage_guide.md` 可能包含示例或真实凭据，后续应单独审计并移除硬编码秘密；不要把其中凭据复制到文档。

### 数据与 Git 噪声

仓库中大量提取 TXT 和生成结果可能已有未提交变更，并包含尾随空格。不要用全仓格式化或清理覆盖这些数据，也不要因 `git diff --check` 的数据文件告警去改动无关内容。验证代码差异时应限定文件范围。

## 下一步建议

1. 用少量用例完成一次付费冒烟评估，确认实时 JSON、全部五类产物和新汇总结构。
2. 评估稳定后再运行 152 题全量评估。
3. 统一 downloader 与 generator/extractor 的默认 JSONL 路径。
4. 将网络测试明确标记为 integration，避免默认 `pytest` 触发 API。

## 常用命令

无网络验证：

```bash
source .venv/bin/activate
python -m pytest -q test_evaluation_logic.py
python -m py_compile evaluation.py gpt-4o-mini_answer.py generate_wikipedia_test_cases.py
```

生成与作答（有 API 成本）：

```bash
python generate_wikipedia_test_cases.py --limit 100
python gpt-4o-mini_answer.py
```

评估（有较高 API 成本）：

```bash
python evaluation.py
```
