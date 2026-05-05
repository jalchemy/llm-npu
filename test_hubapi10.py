import requests

try:
    response = requests.get('https://modelscope.cn/api/v1/models?name_or_path=openvino')
    print("Status:", response.status_code)
    print("Text:", response.text[:200])
except Exception as e:
    print(e)
