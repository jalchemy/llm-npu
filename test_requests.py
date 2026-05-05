import requests
import re
try:
    url = "https://www.modelscope.cn/models?search=openvino"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    matches = re.findall(r'"Name":"([^"]+)"', response.text)
    print(matches[:5])
    matches_id = set()
    for match in re.findall(r'/models/([^/]+/[^/"\'<>]+)', response.text):
        if '?' not in match and '#' not in match:
            matches_id.add(match)
    print(list(matches_id)[:5])
except Exception as e:
    print(e)
