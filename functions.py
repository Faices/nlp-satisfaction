from pandas_profiling import ProfileReport
from dotenv import load_dotenv
import os
import pyodbc
import re


def sql_azure_connect():
    ## Load .env file
    load_dotenv('config/.env')
    
    ## Import credentials for kuzu Azure DB from .env file
    SERVER_AZURE = os.getenv('SERVER_AZURE', "default")  
    DATABASE_AZURE = os.getenv('DATABASE_AZURE', "default")
    USERNAME_AZURE = os.getenv('USERNAME_AZURE', "default")
    PASSWORD_AZURE = os.getenv('PASSWORD_AZURE', "default")
    DRIVER_AZURE = os.getenv('DRIVER_AZURE', "default")
    
    cnxn = pyodbc.connect('DRIVER='+DRIVER_AZURE+';SERVER='+SERVER_AZURE+';DATABASE='+DATABASE_AZURE+';UID='+USERNAME_AZURE+';PWD='+ PASSWORD_AZURE)

    ## Show available tables
    table_names = [x[2] for x in cnxn.cursor().tables(tableType='TABLE')]
    print("Available tables: ",table_names)

    return cnxn

def filter_dateframe_cols(df,cols:list):
    df = df[[cols]]
    
    
def add_basic_textfeatures (df,colname:str):
    '''This function takes as input a dataframe and the column name with the text and adds basic textfeatures as new columns. The functions returns a new dataframe'''
    
    dff = df.copy()
    
     ## Add character count
    charactercount = df[colname].apply(lambda x: len(x))
    dff[colname + '_' + 'Character'] =  charactercount
    
    ## Add token count (wordcount)
    tokencount = df[colname].apply(lambda x: len(str(x).split()))
    dff[colname + '_' + 'Tokens'] = tokencount
    
    ## Add types count (unique wordcount)
    typecount = df[colname].apply(lambda x: len(set(str(x).split())))
    dff[colname + '_' + 'Types'] = typecount
    
    ## Add TTR (Type-Token Ratio)
    dff[colname + '_' + 'TTR'] = (typecount/tokencount)*100
    
    return dff

def remove_redundant_whitespaces(column):
    '''Removes all additional whitespaces from a list ans returns a new list'''
    
    templist = []
    
    for x in column:
        templist.append (re.sub(r'\s+'," ", x).strip())
    
    return templist



    

