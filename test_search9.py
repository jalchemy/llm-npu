import requests

try:
    response = requests.get('https://modelscope.cn/api/v1/models/search?name=openvino')
    print(response.status_code)
except Exception as e:
    print(e)
