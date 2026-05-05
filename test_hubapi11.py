import requests

url = "https://www.modelscope.cn/api/v1/models"
payload = {
    "SortBy": "GmtModified",
    "Order": "Desc",
    "PageNumber": 1,
    "PageSize": 10,
    "Name": "openvino"
}
headers = {
    "User-Agent": "Mozilla/5.0"
}

try:
    # Modelscope search seems to be a GET request based on devtools
    response = requests.get(url, params=payload, headers=headers)
    print("GET Status:", response.status_code)
    # print(response.text[:200])
except Exception as e:
    print(e)
