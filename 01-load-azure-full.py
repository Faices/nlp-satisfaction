import pandas as pd
import timeit
from functions import sql_azure_connect

## connect to azure database
cnxn = sql_azure_connect()

## Define relevant survey
UmfrageName = 'kuzu_zug'

q = f"""SELECT FrageCode FROM Frage WHERE UmfrageName LIKE '{UmfrageName}';""" 
cols =  pd.read_sql(q, con=cnxn)
col_list =  cols.FrageCode.values.tolist()

## Add manual cols of interest
col_list.insert(0, "file_name")
col_list.insert(0, "UmfrageName")
col_list.insert(0, "participant")
col_list.insert(0, "time")
## Join to one list
cols = ', '.join(col_list)

## Bild SQL query
query = f"""SELECT {cols} FROM Teilnehmer WHERE UmfrageName LIKE '{UmfrageName}';"""

## Load data
start = timeit.default_timer()

kuzu_zug =  pd.read_sql(query , con=cnxn)

end = timeit.default_timer()
print("Duration: ",end-start)

## Save file
kuzu_zug.to_feather("data/DataRaw")