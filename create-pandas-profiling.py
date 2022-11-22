from pandas_profiling import ProfileReport
import pandas as pd
import datetime

## import data
filename = "DataRaw"
folderpath = "data/"
df = pd.read_feather(folderpath + filename)


# Pandas Profiling TextData
profile = ProfileReport(
    df,
    title=filename,
    lazy=False,
    dark_mode=True,
    minimal=True,
)

profile.to_file("html/ProfilingDataRaw.html")
