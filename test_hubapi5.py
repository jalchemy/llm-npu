from modelscope.hub.api import HubApi

api = HubApi()
# The issue with modelscope is list_models takes owner_or_group and it only lists models inside that group/user.
# But there must be a way to search. Let's look at huggingface for now.
