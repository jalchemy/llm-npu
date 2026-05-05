import subprocess
print("Attempting to run download logic...")
try:
    subprocess.run([
        "python3", "-c",
        "from modelscope import snapshot_download; snapshot_download('zhaohb/Qwen3-4B-int4-sym-ov-npu', local_dir='models/qwen3_4b_npu')"
    ], check=True)
except Exception as e:
    print("Caught exception:", e)
