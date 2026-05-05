import requests
import json

try:
    headers = {"Content-Type": "application/json"}
    payload = {"SortBy": "GmtModified", "Order": "Desc", "PageNumber": 1, "PageSize": 10, "Name": "openvino"}
    response = requests.post('https://www.modelscope.cn/api/v1/models', json=payload, headers=headers)
    print(response.status_code)
    # print(response.text)
except Exception as e:
    print(e)
