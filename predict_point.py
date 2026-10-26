import pyarrow.parquet as pq
t = pq.read_table(r"D:\IIT BBSR\Suraj Sir work\sm_pipeline\static\source\input.parquet",
                  columns=["Geology", "soil_moisture_2025", "SMAP_SM_2025"]).to_pandas()
print(sorted(t.Geology.unique()))
print(t[["soil_moisture_2025", "SMAP_SM_2025"]].describe())