from modelscope.hub.api import HubApi
api = HubApi()
try:
    print(dir(api))
    models = api.search_model("openvino", limit=5)
    print("MS:", models)
except Exception as e:
    import traceback
    traceback.print_exc()

import inspect
print(inspect.signature(api.list_models))
