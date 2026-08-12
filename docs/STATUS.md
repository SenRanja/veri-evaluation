# 项目状态

更新日期：2026-08-12

## 当前可用状态

- Python 依赖已写入根目录 `requirements.txt`。
- 本地 `.venv` 使用 Python 3.12.13，核心依赖可导入。
- `config.yaml` 当前设置：`target.model=gpt-4o-mini`、`judge.model=gpt-4o-mini`、`evaluation.max_workers=4`、四项阈值均为 0.7。
- Wikipedia 源数据 `evaluation_cases/wikipedia_10000.jsonl` 当前有 4,970 行。
- 当前用例 `evaluation_cases/test_cases_novel.json` 有 4,000 篇文档、16,000 道题；目前全部只有无后缀的 `null` 占位，没有 `gpt-4o-mini` 模型后缀作答字段，因此不能直接评估。
- `evaluation_cases/test_cases_novel - Copy.json` 有 101 篇文档、401 道题，401 道题均已有完整的 `gpt-4o-mini` 模型后缀作答字段；如需评估它，必须同时将 `config.yaml` 的 `project.cases_file` 指向该文件。
- 作答器已改为只使用用例 JSON 中保存的 `retrieval_context`，不再读取外部完整 TXT。
- 已新增 `veriai_answer.py`：按文档顺序上传 TXT 到 Veris，保存 `veri_file_id`，再按题保存 `actual_answered_veri` 和 `actual_output_veri`；上传和回答均逐项原子落盘并支持断点恢复。启动时只索引一次 TXT 目录，不做全量预校验；文档、上传或单题异常会记录后跳过。成功响应不受引用校验或 Boolean 判定阻塞，完整输出会追加 `file_id`、文件名和标题来源索引。
- 当前用例前 10 篇文档已完成 Veris 测试，共保存 10 个文件 ID 和 40 道非空回答；40 道回答均含回答、引用、来源和本地来源索引区块，重复运行 `--document-limit 10` 会跳过全部网络调用。
- 评估逻辑已实现目标/裁判模型分离、模型字段选择、指标适用性、条件质量汇总和 CSV 门控标记。
- `results.json` 现在从运行开始存在，并在每个 metric 响应后原子更新；总体汇总只统计完整用例，每完成一题即时输出实时总体指标。
- 指标门控和汇总已使用稳定内部 ID，不再依赖 DeepEval 可为空的 `name`。
- `test_evaluation_logic.py` 的 8 项离线测试已通过，覆盖每响应落盘、真实指标 ID、门控、字段选择和条件汇总。
- `test_veriai_answer_logic.py` 的 13 项离线测试已通过，覆盖聊天响应解析、中英文拒答判定、宽松引用归一化和来源索引去重。
- 当时也已通过核心 Python 文件编译和 VS Code 诊断检查。
- 已完成一次 152 题全量评估：`evaluation_results/20260812-153056-gpt-4o-mini/` 含全部五类产物，共收到 608 个指标结果；Decision Accuracy 为 93.42%，Correct Answer Rate 为 81.58%。该结果对应之前的 152 题数据快照，不代表当前 16,000 题文件。
- WSL 可直接运行 `bash evaluation.sh`；RackNerd 的 `/root/veri-evaluation` 已建立 Python 3.12 `.venv` 并安装依赖，但同样必须先为当前用例生成目标模型作答。

## 尚未完成

- 当前 16,000 道题尚未运行 `gpt-4o-mini_answer.py`，因此 `evaluation.py` 会在首题校验时报 `must contain Boolean actual_answered`。
- Veris 目前只完成前 10 篇文档；其余文档尚未上传或作答。`config.yaml` 当前目标模型仍是 `gpt-4o-mini`，因此评估器不会读取 `_veri` 字段。
- 当前 16,000 题数据尚未进行全量评估；作答和四指标评估都会产生大量 API 调用、费用和运行时间，应先用 `--limit` 分批作答并检查成本。
- 未运行网络测试 `test_chatbot.py` 与 `test_veris.py`，它们会调用 LLM 并产生费用。

## 已知问题与风险

### 数据路径不一致

`wiki_downloader/wiki_downloader.py` 默认写入 `wiki_downloader/data/wikipedia_10000.jsonl`，而生成器和提取器默认读取 `evaluation_cases/wikipedia_10000.jsonl`。新下载流程需要手动移动/指定路径，或后续统一默认值。

### 成本和稳定性

- `generate_wikipedia_test_cases.py`、`gpt-4o-mini_answer.py`、`evaluation.py` 都会调用 API。
- 完整评估每题运行四项 LLM 指标；当前 16,000 题理论上需要 64,000 个指标结果，成本远高于已完成的 152 题运行。
- 之前以 16 并发运行时留下未完成目录；当前已降为 4。中断时实时 `results.json` 会保留已返回结果，但评估器本身不续跑，重新执行会创建新的时间戳目录并从头评估。
- 作答器每题原子保存，并默认跳过已有的完整模型后缀字段；可通过重复执行安全续跑。不要使用 `--overwrite`，除非明确需要重生成已有回答。
- `test_chatbot.py` 和 `test_veris.py` 是网络测试，其中存在硬编码裁判模型，不应作为默认离线测试运行。

### 日志与凭据

- `openai_interactions.jsonl` 可包含完整问题、上下文和回答。
- `.env` 和任何 Bearer/API token 不得提交。
- `Veris_LLM_usage_guide.md` 可能包含示例或真实凭据，后续应单独审计并移除硬编码秘密；不要把其中凭据复制到文档。

### 数据与 Git 噪声

仓库中大量提取 TXT 和生成结果可能已有未提交变更，并包含尾随空格。不要用全仓格式化或清理覆盖这些数据，也不要因 `git diff --check` 的数据文件告警去改动无关内容。验证代码差异时应限定文件范围。

## 下一步建议

1. 使用 `python gpt-4o-mini_answer.py --limit <小批量>` 为当前数据分批生成回答，确认字段、速率和成本；重复运行会跳过已完成题目。
2. 只有在 `config.yaml` 所指文件中的所有题目都包含完整目标模型后缀字段后，才运行 `bash evaluation.sh`；评估器不接受部分作答文件。
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
python gpt-4o-mini_answer.py --limit 100
# 重复执行会跳过已完成题目；去掉 --limit 可处理全部剩余题目
python gpt-4o-mini_answer.py
```

评估（有较高 API 成本）：

```bash
bash evaluation.sh
```

WSL 与 RackNerd 使用相同顺序：先进入项目目录并确保 `.env` 中有 `OPENAI_API_KEY`，再运行作答器，全部回答完成后运行 `bash evaluation.sh`。WSL 项目目录当前为 `/home/jayd/code/veri-evaluation`；RackNerd 项目目录当前为 `/root/veri-evaluation`。
