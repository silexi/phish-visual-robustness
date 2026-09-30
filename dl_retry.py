import pandas as pd, os, time
base="https://huggingface.co/api/datasets/nyuuzyou/phishing-snapshots/parquet/default/train"
for i in range(1, 20):
    f=f"shard{i}.parquet"
    if os.path.exists(f): continue
    for attempt in range(4):
        try:
            df=pd.read_parquet(f"{base}/{i}.parquet")
            keep=df[(df.has_password_field)&(df.http_status==200)&(df.html_length>3000)]
            keep.to_parquet(f); print(f"shard{i}: {len(keep)} aday", flush=True); break
        except Exception as e:
            print(f"shard{i} deneme {attempt+1} basarisiz: {type(e).__name__}", flush=True)
            time.sleep(8)
print("DONE")
