from functions import sql_azure_connect
import pandas as pd
import datetime

## connect to azure database
cnxn = sql_azure_connect()

######################################## full load ################################################
## Define relevant survey
UmfrageName = 'kuzu_zug'

q = f"""SELECT FrageCode FROM Frage WHERE UmfrageName LIKE '{UmfrageName}';""" # We want all FrageCodes here
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
starttime = datetime.datetime.now()
print("Start Loading data from Azure: ",starttime)

kuzu_zug =  pd.read_sql(query , con=cnxn)

endtime = datetime.datetime.now()
print("finished Loading data from Azure: ",endtime)
print("Duration: ",endtime-starttime)

## Save file
kuzu_zug.to_feather("data/DataRaw") # store data in feather file


##################################################################################################