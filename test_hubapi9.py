from modelscope.hub.api import HubApi

api = HubApi()
# In modelscope SDK, how does it search? It doesn't seem to support search across all models.
# I will use huggingface_hub for both searching HuggingFace AND maybe we don't need ModelScope search if not possible?
# The prompt says: "search Modelscope and Huggingface for models that meet the keywords associated with the search if a 'Search model repos' option is chosen"
