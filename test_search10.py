from modelscope.hub.api import HubApi
import json

api = HubApi()
response = api.session.get('https://modelscope.cn/api/v1/models?Name=openvino')
print(response.status_code)
