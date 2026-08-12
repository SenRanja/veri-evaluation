

# 复现

## 前置准备：生成 UUID

    CHAT_ID=$(uuidgen) && MSG_ID=$(uuidgen) && echo "chat_id: $CHAT_ID" && echo "message_id: $MSG_I

chat_id: 4e30b667-9ea4-4d97-8b15-33c35c3ab31f
message_id: 46734fa7-5f38-4e24-98be-10c471a0eb89

## 第一步：上传文件

info.txt:
```txt
X
真实姓名：Raj Eric
Username @RajEric212166
+85252112504  (临时租借平台弄得，非长期、真实手机号)
May 1st 1995

邮箱：访问httpsnavigator-lxa.mail.com，注册信息同上虚拟身份
beijixing012@mail.com
qwer!@#$QWER


```

### info.txt "id":"283904c7-bb3d-4ab2-ac0f-c0080cce48fc"

```bash
curl -X POST 'https://verisai.duckdns.org/api/v1/files/' \
  -H 'Authorization: Bearer sk-c863e7df515340c4a3c450f74bdc23e7' \
  -F 'file=@info.txt' \
  -F 'process=true'
```

返回
```json
{
  "id":"283904c7-bb3d-4ab2-ac0f-c0080cce48fc",
  "user_id":"ba625d43-bab0-41cb-a829-72f48ba39843",
  "hash":"6ac2e6fad9aff4f1a21a0f9e75fce821f046f38ccabf045ddd2e5ab843e5f8db",
  "filename":"info.txt",
  "data":{
    "content":"X\n真实姓名:Raj Eric\nUsername @RajEric212166\n+85252112504  (临时租借平台弄得,非长期、真实手机号)\nMay 1st 1995\n\n邮箱:访问httpsnavigator-lxa.mail.com,注册信息同上虚拟身份\nbeijixing012@mail.com\nqwer!@#$QWER\n\n"},
    "meta":{
      "name":"info.txt",
      "content_type":"text/plain",
      "size":257,
      "data":{}
      },
    "created_at":1784803679,
    "updated_at":1784803679
}
```

### santi.txt "id":"7d99fec3-2421-4da1-948e-22d77dd2ed27"

上传三体3的前言片段后返回下面文件id。

```
{"id":"7d99fec3-2421-4da1-948e-22d77dd2ed27","user_id":"ba625d43-bab0-41cb-a829-72f48ba39843","hash":"0c7bb81f80507f777ac4a4ae25e64bfe66005d6766c79d5da68f5f2bc71fd710","filename":"santi.txt","data":{"content":"三体Ⅲ·死神永生 刘慈欣
```


## 第二步：提问（带 true_meaning_search）

```bash
NEW_MSG_ID=$(uuidgen)

curl -X POST 'https://verisai.duckdns.org/api/chat/completions' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-c863e7df515340c4a3c450f74bdc23e7' \
  -d '{
    "model": "arena-model",
    "messages": [
      {"role": "user", "content": "这个人叫什么"},
      {"role": "assistant", "content": "他目前几岁了？"},
      {"role": "user", "content": "他的电话和邮箱是什么？"}
    ],
    "features": {"true_meaning_search": true},
    "files": [{"id": "283904c7-bb3d-4ab2-ac0f-c0080cce48fc", "type": "file"}],
    "chat_id": "4e30b667-9ea4-4d97-8b15-33c35c3ab31f",
    "id": "$NEW_MSG_ID"
  }'
```

返回

```
"我幫您看了一下資料，他的電話號碼是 +85252112504（不過這似乎是臨時租借的號碼，不是長期真實手機號哦）。至於郵箱，是 beijixing012@mail.com。希望這些資訊對您有幫助！😊"
```



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
      {"role": "user", "content": "这个人住在哪里？"},
      {"role": "assistant", "content": "告诉我这个人叫什么"},
      {"role": "user", "content": "告诉我他全部的邮箱"}
    ],
    "features": {"true_meaning_search": true},
    "files": [{"id": "283904c7-bb3d-4ab2-ac0f-c0080cce48fc", "type": "file"}],
    "chat_id": "4e30b667-9ea4-4d97-8b15-33c35c3ab31f",
    "id": "$NEW_MSG_ID"
  }'
```


