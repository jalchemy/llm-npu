from modelscope.hub.api import HubApi
import inspect

api = HubApi()
for method in ['list_models']:
    print(inspect.getsource(getattr(api, method)))
