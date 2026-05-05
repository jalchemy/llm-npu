import requests

try:
    url = "https://modelscope.cn/api/v1/models"
    params = {
        "Name": "openvino",
        "PageNumber": 1,
        "PageSize": 10
    }
    response = requests.get(url, params=params)
    print(response.status_code)
    if response.status_code == 200:
        print("Success!")
except Exception as e:
    print(e)
