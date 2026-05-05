from modelscope.hub.api import HubApi

api = HubApi()
# What happens if we pass empty owner_or_group ?
try:
    models = api.list_models(owner_or_group="", page_size=10)
    print("Models count:", models['TotalCount'])
    for m in models['Models'][:2]:
        print(m['Name'])
except Exception as e:
    print(e)
