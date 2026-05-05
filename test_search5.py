from modelscope.hub.api import HubApi
api = HubApi()
try:
    models = api.list_models(search="openvino", page_size=5)
    print([m["Id"] for m in models["Models"]])
except Exception as e:
    import traceback
    traceback.print_exc()

import requests
try:
    # See if modelscope has an open api
    response = requests.get('https://modelscope.cn/api/v1/models?name=openvino')
    print([m["Id"] for m in response.json()["Data"]["Models"][:5]])
except Exception as e:
    print("REST:", e)
