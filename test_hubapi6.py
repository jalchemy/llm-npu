import requests

try:
    headers = {"Content-Type": "application/json"}
    payload = {"Name": "openvino"}
    response = requests.get('https://modelscope.cn/api/v1/models?Name=openvino', headers=headers)
    print(response.status_code)
except Exception as e:
    print(e)
