from modelscope import snapshot_download
import os

try:
    print(f"Current dir: {os.getcwd()}")
    snapshot_download('zhaohb/Qwen3-4B-int4-sym-ov-npu', local_dir='models/qwen3_4b_npu')
except Exception as e:
    import traceback
    traceback.print_exc()
