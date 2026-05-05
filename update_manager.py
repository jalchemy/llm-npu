import re

with open('manager.py', 'r') as f:
    content = f.read()

# Add HuggingFace and ModelScope search
search_code = """
def search_models(keyword):
    results = []

    # 1. Search HuggingFace
    print(f"Searching HuggingFace for '{keyword}'...")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        hf_models = api.list_models(search=keyword, limit=5)
        for m in hf_models:
            results.append({"id": m.id, "source": "huggingface"})
    except Exception as e:
        print(f"Error searching HuggingFace: {e}")

    # 2. Search ModelScope
    print(f"Searching ModelScope for '{keyword}'...")
    try:
        from modelscope.hub.api import HubApi
        import json
        api = HubApi()
        cookies = api.get_cookies(access_token=None, cookies_required=False)
        path = f'{api.endpoint}/api/v1/models/'
        payload = {"Name": keyword, "PageNumber": 1, "PageSize": 5}
        r = api.session.put(
            path,
            data=json.dumps(payload),
            cookies=cookies,
            headers=api.builder_headers(api.headers))
        if r.status_code == 200:
            ms_models = r.json().get('Data', {}).get('Models', [])
            for m in ms_models:
                results.append({"id": f"{m['Path']}/{m['Name']}", "source": "modelscope"})
    except Exception as e:
        print(f"Error searching ModelScope: {e}")

    return results
"""
