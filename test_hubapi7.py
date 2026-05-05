import requests

try:
    response = requests.get('https://www.modelscope.cn/api/v1/models?name_or_path=openvino')
    print(response.status_code)
except Exception as e:
    print(e)
