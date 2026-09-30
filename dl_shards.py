import pandas as pd, sys, time
base="https://huggingface.co/api/datasets/nyuuzyou/phishing-snapshots/parquet/default/train"
for i in range(1, 12):
    t=time.time()
    df=pd.read_parquet(f"{base}/{i}.parquet")
    keep=df[(df.has_password_field)&(df.http_status==200)&(df.html_length>3000)]
    keep.to_parquet(f"shard{i}.parquet")
    print(f"shard{i}: {len(keep)}/{len(df)} aday, {time.time()-t:.0f}s", flush=True)
print("DONE")
