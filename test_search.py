from huggingface_hub import HfApi
try:
    api = HfApi()
    models = api.list_models(search="openvino", limit=5)
    for m in models:
        print("HF:", m.id)
except Exception as e:
    print("HF err:", e)

try:
    from modelscope.hub.api import HubApi
    api = HubApi()
    models = api.list_models(name_or_path="openvino")
    for m in models["Models"][:5]:
        print("MS:", m["Path"])
except Exception as e:
    import traceback
    traceback.print_exc()
