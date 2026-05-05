from modelscope.hub.api import HubApi

api = HubApi()
# According to https://github.com/modelscope/modelscope/blob/master/modelscope/hub/api.py
# There is a `search_model` ? Wait, it threw AttributeError. Let's list all methods of HubApi again.
print([m for m in dir(api) if not m.startswith('_')])
