from modelscope.hub.api import HubApi

api = HubApi()
# wait, list_models uses a PUT request internally, with `data='{"Path":"%s", "PageNumber":%s, "PageSize": %s}' % (owner_or_group, page_number, page_size)`.
# Since there's no native method to search all models by keyword in HubApi, what if we just list all models and filter? It's 194014 models! We can't do that.
# Let's see if we can pass "Name" instead of "Path" in the PUT payload if we override it?
import inspect
print(inspect.getsource(api.list_models))
