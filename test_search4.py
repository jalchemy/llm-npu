import requests

try:
    response = requests.get('https://modelscope.cn/api/v1/models?Name=openvino')
    print(response.text[:200])
except Exception as e:
    import traceback
    traceback.print_exc()
