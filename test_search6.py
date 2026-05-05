import requests

try:
    response = requests.get('https://modelscope.cn/api/v1/models', params={'name_or_path': 'openvino', 'page_size': 5})
    print(response.json())
except Exception as e:
    print(e)
