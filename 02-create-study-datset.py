import pandas as pd
from functions import filter_dateframe_cols
import openpyxl
import numpy as np
import re
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
df = df[(df['u_date'] >= "2019-01-01") & (df['u_date'] < "2022-12-31")]

# filter language
df = df[df['S_sprache']=="Deutsch"] #Aufgrund der Auswahlkriterien werden nur Deutschspraige Records verwendet

# filter language
df = df[df['S_sprache']=="Deutsch"] #Aufgrund der Auswahlkriterien werden nur Deutschspraige Records verwendet



################### Cleaning 1: Replace invalid and missing values with np.nan ###################
## convert empty invalid anwers to np.nan

nan_replace = ['-66','-99','-77','undefined','weiss nicht','keine Zuordnung / GA-Besitzer','weiss nicht / beide Klassen']
df = df.replace(nan_replace , np.nan)
df = df.fillna(value = np.nan)


## convert empty invalid anwers to np.nan (varibale specific)
df['wime_gesamtzuf'] = df['wime_gesamtzuf'].replace(['99','99.0',99,99.0], np.nan) # timewise 99 in statisfacion "wime_gesamtzuf" spotted
df['S_alter'] = df['S_alter'].replace(['99','99.0',99,99.0], np.nan) # eda showd that this is used as placeholder
df['S_sex'] = df['S_sex'].replace("0", np.nan) # mistake in raw data
df['u_klassencode'] = df['u_klassencode'].replace("0", np.nan)  # mistake in raw data
df['R_anschluss'] = df['R_anschluss'].replace("0", np.nan)  # mistake in raw data
df['R_stoerung'] = df['R_stoerung'].replace("Weiss nicht", np.nan)  # same meaning than nan
df['u_ticket'] = df['u_ticket'].replace("3", np.nan)   # mistake in raw data
df['R_zweck'] = df['R_zweck'].replace("0", np.nan)   # mistake in raw data


################### Convert Datatypes ###################


### Manual based on "config.xlsx" ###

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


### Automatic with "convert-dtypes" ###

#df = df.convert_dtypes()


################### Cleaning 2: Replace values with same meaning and harmonize dataset ###################


## clean "S_AB3_HTA": Fix differnet ways of labeling to make variables consistent
df['S_AB3_HTA'] = df['S_AB3_HTA'].replace('quoted', 'ja') # consistent with other var structure
df['S_AB3_HTA'] = df['S_AB3_HTA'].replace('not quoted', 'nein') #consistent with other var structure

## clean "R_zweck": Merging Groups with same meaning
df['R_zweck'] = df['R_zweck'].replace('Freizeitfahrt/ private Ferienreise/ alltägliche Erledigungen (z.B. Arztbesuch, Einkaufen, jmd. Abhol', 'Freizeit und Unterhaltung')
df['R_zweck'] = df['R_zweck'].replace('Freizeitfahrt/ private Ferienreise\r\n', 'Freizeit und Unterhaltung')
df['R_zweck'] = df['R_zweck'].replace('Freizeitfahrt ohne Übernachtung (Ausflug, Kino, Sport, Besuch, usw.)', 'Freizeit und Unterhaltung')
df['R_zweck'] = df['R_zweck'].replace('Geschäftsreise', 'Arbeit und Lernen')
df['R_zweck'] = df['R_zweck'].replace('Fahrt vom oder zum Arbeits-/ Ausbildungsort', 'Arbeit und Lernen')
df['R_zweck'] = df['R_zweck'].replace('Private Ferienreise (Reise mit mind. 1 Übernachtung)', 'Freizeit und Unterhaltung')
df['R_zweck'] = df['R_zweck'].replace('Fahrt zum Arbeitsort', 'Arbeit und Lernen')
df['R_zweck'] = df['R_zweck'].replace('Private Ferienreise (Reise mit mind. 1 Übernachtung)', 'Freizeit und Unterhaltung')
df['R_zweck'] = df['R_zweck'].replace('Alltägliche Erledigungen (z.B. Arztbesuch, Einkaufen, jmd. abholen)', 'Sonstige')
df['R_zweck'] = df['R_zweck'].replace('alltägliche Erledigungen (z.B. Arztbesuch, Einkaufen, jmd. Abholen)\r\n', 'Sonstige')
df['R_zweck'] = df['R_zweck'].replace('Weiss nicht', 'Sonstige')
df['R_zweck'] = df['R_zweck'].replace('Fahrt zum Arbeitsort / Ausbildungsort', 'Arbeit und Lernen')
df['R_zweck'] = df['R_zweck'].replace('Fahrt zum Ausbildungsort', 'Arbeit und Lernen')
df['R_zweck'] = df['R_zweck'].replace('alltägliche Erledigungen (z.B. Arztbesuch, Einkaufen, jmd. Abholen)', 'Freizeit und Unterhaltung')
df['R_zweck'] = df['R_zweck'].replace('Freizeitfahrt/ private Ferienreise', 'Freizeit und Unterhaltung')
df['R_zweck'] = df['R_zweck'].replace('Freizeitfahrt  ohne Übernachtung (Ausflug, Kino, Sport, Besuch, usw.)', 'Freizeit und Unterhaltung')


## clean "u_fahrausweiss": naming
df['u_fahrausweis'] = df['u_fahrausweis'].replace('normales Billett', 'Normales Billett')


###################  Annonymisation ###################

def sent_to_words(sentences):
    for sent in sentences:
        if isinstance(sent, str):# remove emails
            sent = re.sub("(([\w-]+(?:\.[\w-]+)*)@((?:[\w-]+\.)*\w[\w-]{0,66})\." \
                  "([a-z]{2,6}(?:\.[a-z]{2})?))(?![^<]*>)", "<EMAIL>", sent)
            # remove phonenumber
            sent = re.sub(r'/(\b(0041|0)|\B\+41)(\s?\(0\))?(\s)?[1-9]{2}(\s)?[0-9]{3}(\s)?[0-9]{2}(\s)?[0-9]{2}\b/', '<PHONENUMBER>', sent)
            sent = re.sub(r' ([0-9]{10}) ', '<PHONENUMBER>', sent)
            sent = re.sub(r' ([0-9]{13}) ', '<PHONENUMBER>', sent)
            sent = re.sub(r'([0-9]{3}) ([0-9]{3}) ([0-9]{2}) ([0-9]{2})', '<PHONENUMBER>', sent)
            sent = re.sub(r'([0-9]{3})-([0-9]{3})-([0-9]{2})-([0-9]{2})', '<PHONENUMBER>', sent)
            sent = re.sub('\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]?','<PHONENUMBER>',sent)
            sent = re.sub('/(\b(0041|0)|\B\+41)(\s?\(0\))?(\s)?[1-9]{2}(\s)?[0-9]{3}(\s)?[0-9]{2}(\s)?[0-9]{2}\b/','<PHONENUMBER>',sent)
            sent = re.sub('^(\+?)(\d{2,4})(\s?)(\-?)((\(0\))?)(\s?)(\d{2})(\s?)(\-?)(\d{3})(\s?)(\-?)(\d{2})(\s?)(\-?)(\d{2})','<PHONENUMBER>',sent)
            sent = re.sub('/^(?:(?:|0{1,2}|\+{0,2})41(?:|\(0\))|0)([1-9]\d)(\d{3})(\d{2})(\d{2})$/','<PHONENUMBER>',sent)
            sent = re.sub('^(?:\s*-*\s*\d){10}$','<PHONENUMBER>',sent)
            #sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
        else:
            sent = np.NaN
        
        yield(sent)
        
df['Kommentar'] = list(sent_to_words(df.Kommentar.values.tolist()))



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










