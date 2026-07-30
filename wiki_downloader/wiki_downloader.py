import json
import time
from pathlib import Path

import requests

TARGET_COUNT = 10_000
BATCH_SIZE = 10

API_URL = "https://en.wikipedia.org/w/api.php"

OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "wikipedia_10000.jsonl"

HEADERS = {
    # 最好把邮箱替换成你自己的邮箱
    "User-Agent": "sarify/1.0"
}

session = requests.Session()
session.headers.update(HEADERS)


def load_downloaded_ids():
    """读取已经下载的数据，支持中断后继续下载。"""
    downloaded_ids = set()

    if not OUTPUT_FILE.exists():
        return downloaded_ids

    with OUTPUT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                article = json.loads(line)
                downloaded_ids.add(article["page_id"])
            except (json.JSONDecodeError, KeyError):
                continue

    return downloaded_ids


def request_articles():
    params = {
        "action": "query",
        "generator": "random",
        "grnnamespace": 0,           # 只获取正式百科文章
        "grnfilterredir": "nonredirects",
        "grnlimit": BATCH_SIZE,
        "prop": "extracts|info",
        "explaintext": True,         # 返回纯文本
        "inprop": "url",
        "format": "json",
        "formatversion": 2,
        "maxlag": 5,
    }

    response = session.get(
        API_URL,
        params=params,
        timeout=60,
    )

    print("HTTP status:", response.status_code)
    response.raise_for_status()

    return response.json()


downloaded_ids = load_downloaded_ids()
downloaded_count = len(downloaded_ids)

print(f"已经下载：{downloaded_count} 篇")
print(f"保存位置：{OUTPUT_FILE}")

with OUTPUT_FILE.open("a", encoding="utf-8") as output:
    while downloaded_count < TARGET_COUNT:
        try:
            data = request_articles()

            if "error" in data:
                print("API 返回错误：", data["error"])
                time.sleep(10)
                continue

            pages = data.get("query", {}).get("pages", [])

            for page in pages:
                page_id = page.get("pageid")
                title = page.get("title")
                text = page.get("extract", "").strip()

                if not page_id or page_id in downloaded_ids:
                    continue

                # 排除内容非常短的页面
                if len(text) < 200:
                    continue

                article = {
                    "page_id": page_id,
                    "title": title,
                    "text": text,
                    "url": page.get("fullurl"),
                }

                output.write(
                    json.dumps(article, ensure_ascii=False) + "\n"
                )
                output.flush()

                downloaded_ids.add(page_id)
                downloaded_count += 1

                print(
                    f"[{downloaded_count}/{TARGET_COUNT}] "
                    f"{title}"
                )

                if downloaded_count >= TARGET_COUNT:
                    break

            # 避免请求过于频繁
            time.sleep(1)

        except requests.RequestException as error:
            print("网络请求失败：", error)
            print("10 秒后重试……")
            time.sleep(10)

        except json.JSONDecodeError as error:
            print("Wikipedia 返回的不是 JSON：", error)
            print("10 秒后重试……")
            time.sleep(10)

print("下载完成！")
print("文件位置：", OUTPUT_FILE)