# Veris LLM 测试示例

## 前置准备：生成 UUID

```bash
# 方法 1：命令行（推荐）
CHAT_ID=$(uuidgen) && MSG_ID=$(uuidgen) && echo "chat_id: $CHAT_ID" && echo "message_id: $MSG_ID"

# 方法 2：Python
python3 -c "import uuid; print(f'chat_id: {uuid.uuid4()}'); print(f'message_id: {uuid.uuid4()}')"
```

记下输出的两个 UUID，后续命令中会用到。

---

## 第一步：上传文件

```bash
curl -X POST 'https://verisai.duckdns.org/api/v1/files/' \
  -H 'Authorization: Bearer sk-c863e7df515340c4a3c450f74bdc23e7' \
  -F 'file=@琼瑶的新闻.txt' \
  -F 'process=true'
```

**预期返回**：
```json
{"id":"xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", ...}
```

记下返回的 `id`（文件 ID），例如：`850a1e85-824d-4214-83f4-83b93419a158`

---

## 第二步：提问（带 true_meaning_search）

```bash
curl -X POST 'https://verisai.duckdns.org/api/chat/completions' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-c863e7df515340c4a3c450f74bdc23e7' \
  -d '{
    "model": "arena-model",
    "messages": [{"role": "user", "content": "琼瑶几岁？"}],
    "features": {"true_meaning_search": true},
    "files": [{"id": "第一步返回的文件ID", "type": "file"}],
    "chat_id": "第二步中生成的chat_id",
    "id": "第二步中生成的message_id"
  }'
```

**预期返回**：
- 成功：`"琼瑶享年86岁。🕊️"`
- 文件无相关内容：`"（无相关 segment）..."`
- 文件不存在：`"真意查询失败"`

---

## 同一会话继续提问

同一个会话不需要重新生成 `chat_id`，只需生成新的 `message_id`：

```bash
NEW_MSG_ID=$(uuidgen)

curl -X POST 'https://verisai.duckdns.org/api/chat/completions' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-c863e7df515340c4a3c450f74bdc23e7' \
  -d '{
    "model": "arena-model",
    "messages": [
      {"role": "user", "content": "琼瑶几岁？"},
      {"role": "assistant", "content": "琼瑶享年86岁。🕊️"},
      {"role": "user", "content": "她有哪些代表作？"}
    ],
    "features": {"true_meaning_search": true},
    "files": [{"id": "文件ID", "type": "file"}],
    "chat_id": "相同的chat_id",
    "id": "$NEW_MSG_ID"
  }'
```

---

## 字段说明

| 字段 | 说明 |
|------|------|
| `chat_id` | 会话 ID，同一会话复用，每次新会话需生成新的 UUID |
| `id`（message_id） | 消息 ID，每条消息需生成新的 UUID |
| `chat_id` 和 `id` | 由客户端生成，系统不自动生成 |

## api key

sk-c863e7df515340c4a3c450f74bdc23e7
