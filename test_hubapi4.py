import requests

try:
    headers = {"Content-Type": "application/json"}
    payload = {"Name": "openvino"}
    response = requests.post('https://modelscope.cn/api/v1/models', json=payload, headers=headers)
    print(response.status_code)
except Exception as e:
    print(e)
