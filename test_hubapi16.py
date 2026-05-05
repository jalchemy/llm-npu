from modelscope.hub.api import HubApi
import json
api = HubApi()

cookies = api.get_cookies(access_token=None, cookies_required=False)
path = f'{api.endpoint}/api/v1/models/'
payload = {
    "Name": "openvino",
    "PageNumber": 1,
    "PageSize": 10
}
r = api.session.put(
    path,
    data=json.dumps(payload),
    cookies=cookies,
    headers=api.builder_headers(api.headers))

print(r.status_code)
if r.status_code == 200:
    print(r.json())
