import pandas as pd
from code.functions import *
import numpy as np
import datetime

################### Timer ###################
starttime = datetime.datetime.now()
print("Start: ",starttime)


################### Import ###################
filelocation = 'data/DataRaw'
df = pd.read_feather(filelocation)

## load config file
config = pd.read_excel('config/config.xlsx',sheet_name='fragecodes')


################### Filter ###################
## filter columns
df = df[config['fragecode']]

# filter date range
df['u_date'] = pd.to_datetime(df['u_date'])
df_1 = df[(df['u_date'] >= "2019-01-01") & (df['u_date'] < "2020-04-01")] # ignore the stoped 2 COVID-19 Month  april,mai 2020
df_2 = df[(df['u_date'] >= "2020-06-01") & (df['u_date'] <= "2022-12-31")] # ignore the stoped 2 COVID-19 Month  april,mai 2020
df = pd.concat([df_1, df_2],ignore_index=True)

## print len all languages
print(len(df))

# filter language
df = df[df['S_sprache']=="Deutsch"] #Aufgrund der Auswahlkriterien werden nur Deutschspraige Records verwendet



################### Cleaning 1: Replace invalid and missing values with np.nan ###################
## convert empty invalid anwers to np.nan

nan_replace = ['-66','-99','-77','undefined','weiss nicht','keine Zuordnung / GA-Besitzer','weiss nicht / beide Klassen']
df = df.replace(nan_replace , np.nan)
df = df.fillna(value = np.nan)

nan_replace_variables = {'wime_gesamtzuf': ['99','99.0',99,99.0], 
                         'S_alter': ['99','99.0',99,99.0], 
                         'S_sex': "0", 
                         'u_klassencode': "0", 
                         'R_anschluss': "0", 
                         'R_stoerung': "Weiss nicht", 
                         'u_ticket': "3", 
                         'R_zweck': "0"}
for variable, replacement in nan_replace_variables.items():
    df[variable] = df[variable].replace(replacement, np.nan)


################### Convert Datatypes ###################


# ### Manual based on "config.xlsx" ###

# Convert to Categorical
tocategory = config[config['datatype'] == 'category']
tocategory = tocategory['fragecode']
for item in tocategory:
    df[item] = df[item].astype('category')


# Convert to date
todate = config[config['datatype'] == 'datetime']
todate = todate['fragecode']
for item in todate:
    df[item] = pd.to_datetime(df[item])


# Convert to time
totime = config[config['datatype'] == 'time']
totime = totime['fragecode']
for item in totime:
    df[item] = pd.to_datetime(df[item],format='%H:%M:%S').dt.time


# convert to string
tostring = config[config['datatype'] == 'string']
tostring = tostring['fragecode']
for item in tostring:
    df[item] = df[item].astype('string')
    
    
# convert to numeric
tonumeric = config[config['datatype'] == 'numeric']
tonumeric = tonumeric['fragecode']
for item in tonumeric:
    df[item] = pd.to_numeric(df[item])


df = df.reset_index(drop=True)



################### Cleaning 2: Replace values with same meaning and harmonize dataset ###################


## clean "S_AB3_HTA": Fix differnet ways of labeling to make variables consistent
df['S_AB3_HTA'] = df['S_AB3_HTA'].replace('quoted', 'ja') # consistent with other var structure
df['S_AB3_HTA'] = df['S_AB3_HTA'].replace('not quoted', 'nein') #consistent with other var structure

## clean "R_zweck": Merging Groups with same meaning
replacements = {
    'Freizeitfahrt/ private Ferienreise/ alltägliche Erledigungen (z.B. Arztbesuch, Einkaufen, jmd. Abhol': 'Freizeit und Unterhaltung',
    'Freizeitfahrt/ private Ferienreise\r\n': 'Freizeit und Unterhaltung',
    'Freizeitfahrt ohne Übernachtung (Ausflug, Kino, Sport, Besuch, usw.)': 'Freizeit und Unterhaltung',
    'Geschäftsreise': 'Arbeit und Lernen',
    'Fahrt vom oder zum Arbeits-/ Ausbildungsort': 'Arbeit und Lernen',
    'Private Ferienreise (Reise mit mind. 1 Übernachtung)': 'Freizeit und Unterhaltung',
    'Fahrt zum Arbeitsort': 'Arbeit und Lernen',
    'Alltägliche Erledigungen (z.B. Arztbesuch, Einkaufen, jmd. abholen)': 'Sonstige',
    'alltägliche Erledigungen (z.B. Arztbesuch, Einkaufen, jmd. Abholen)\r\n': 'Sonstige',
    'Weiss nicht': 'Sonstige',
    'Fahrt zum Arbeitsort / Ausbildungsort': 'Arbeit und Lernen',
    'Fahrt zum Ausbildungsort': 'Arbeit und Lernen',
    'alltägliche Erledigungen (z.B. Arztbesuch, Einkaufen, jmd. Abholen)': 'Freizeit und Unterhaltung',
    'Freizeitfahrt/ private Ferienreise': 'Freizeit und Unterhaltung',
    'Freizeitfahrt  ohne Übernachtung (Ausflug, Kino, Sport, Besuch, usw.)': 'Freizeit und Unterhaltung',
}

df['R_zweck'] = df['R_zweck'].replace(replacements)


## clean "u_fahrausweiss": naming
df['u_fahrausweis'] = df['u_fahrausweis'].replace('normales Billett', 'Normales Billett')


################### Tranform Staisfaction Questions ###################
# Transform from 5 and 10 scaling to 1harmonized 100 scale (10 until 03.2020 and 5 since 5.2020 - april 2020 no data)
satisfaction_questions = config[config['vartype'] == 'satisfaction']
satisfaction_questions = list(satisfaction_questions['fragecode']) #safe satisfaction codes in list

mask_10 = (df['u_date'] <= '2020-4-30')
mask_5 = (df['u_date'] >= '2020-5-1')
df_satisfaction_10 = df.loc[mask_10]
df_satisfaction_5 = df.loc[mask_5]

# Transform from 10 to 100 scale
for item in satisfaction_questions:
    df_satisfaction_10[item] = df_satisfaction_10[item].apply(lambda x:(x-1)/9*100)
    
# Transform from 5 to 100 scale
for item in satisfaction_questions:
    df_satisfaction_5[item] = df_satisfaction_5[item].apply(lambda x:(x-1)/4*100)
    
df = pd.concat([df_satisfaction_10, df_satisfaction_5])



############## Export ############## 
df = df.reset_index(drop=True)
df.to_feather('data/DataClean') # store data in feather file


################### Timer ###################
endtime = datetime.datetime.now()
print("End: ",endtime)
print("Duration: ",endtime-starttime)










