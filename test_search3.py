import requests

try:
    response = requests.get('https://modelscope.cn/api/v1/models?name_or_path=openvino')
    print(response.json())
except Exception as e:
    import traceback
    traceback.print_exc()
