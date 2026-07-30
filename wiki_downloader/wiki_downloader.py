from pathlib import Path
import requests

titles = [
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
]

# 保存到当前脚本旁边的 articles 文件夹
output_dir = Path(__file__).resolve().parent / "articles"
output_dir.mkdir(exist_ok=True)

url = "https://en.wikipedia.org/w/api.php"

headers = {
    "User-Agent": "WikiArticleDownloader/1.0 (personal educational project)"
}

for title in titles:
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "redirects": True,
        "titles": title,
        "format": "json",
        "formatversion": 2,
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=30,
        )

        print(f"{title}: HTTP {response.status_code}")
        response.raise_for_status()

        data = response.json()
        page = data["query"]["pages"][0]

        if "missing" in page:
            print(f"找不到文章：{title}")
            continue

        filename = title.replace(" ", "_") + ".txt"
        file_path = output_dir / filename
        file_path.write_text(page.get("extract", ""), encoding="utf-8")

        print(f"下载成功：{file_path}")

    except requests.RequestException as error:
        print(f"请求失败：{title}，原因：{error}")

    except ValueError:
        print("Wikipedia 返回的不是 JSON：")
        print(response.text[:500])