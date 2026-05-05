import requests

try:
    response = requests.get('https://www.modelscope.cn/api/v1/models?Name=openvino')
    print(response.status_code)
except Exception as e:
    print(e)
