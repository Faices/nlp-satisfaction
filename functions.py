from pandas_profiling import ProfileReport
from dotenv import load_dotenv
import os
import pyodbc
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
import pandas as pd


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

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def anonymizer_de(list):
    """
    Anonymize a list of text comments.
    
    This function uses a natural language processing (NLP) engine and an anonymizer engine to identify and anonymize personal information in a list of text comments. The function supports German language comments, and can recognize entities such as person names, email addresses, phone numbers, credit card numbers, and IBAN codes.
    
    Args:
        list: A list of strings containing text comments to be anonymized.
    
    Returns:
        A list of strings containing the anonymized text comments.
    """

    
    
    #Create configuration containing engine name and models
    configuration = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "de", "model_name": "de_core_news_lg"}],}
    
    # Create NLP engine based on configuration
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()
    
    # the languages are needed to load country-specific recognizers 
    # # for finding phones, passport numbers, etc.
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine,
                              supported_languages=["de"])

    # Create the anonymizer
    anonymizer = AnonymizerEngine()

    # Anonymize each comment in the list
    anonymized_text_list = [
        anonymizer.anonymize(text=comment,
                             analyzer_results=analyzer.analyze(text=comment,
                                                              language='de',
                                                              entities=["PERSON","EMAIL_ADDRESS","PHONE_NUMBER","CREDIT_CARD","IBAN_CODE"])).text
        if isinstance(comment, str) else None
        for comment in list
    ]
    
    return anonymized_text_list


def find_extract_ner_entities_list(lst, entitie):
    """
    Find and extract named entities from a list of strings using natural language processing.
    
    This function uses a specified NLP engine and language model to detect named entities in a list of strings,
    and returns a list of the detected entities. If the input is not a string, None is returned instead.
    
    Args:
        lst: A list of strings to process.
        entitie: The named entity type to look for (e.g. "person", "organization", "location", etc.)
        
    Returns:
        A list of detected entities, or None if the input is not a string.
    """

    # Create configuration containing engine name and models
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "de", "model_name": "de_core_news_lg"}],
    }
    
    # Create NLP engine based on configuration
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()
    
    # the languages are needed to load country-specific recognizers 
    # # for finding phones, passport numbers, etc.
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["de"])
    
    # Create a list comprehension that processes each string in the input list and
    # appends the detected entities to the results list. If the input is not a string,
    # None is appended to the results list.
    results_list = [
        None if not isinstance(comment, str)
        else [(comment[res.start:res.end]) for res in analyzer.analyze(text=comment, language="de", entities=[entitie])]
        for comment in lst
    ]

    # Replace empty lists in the results list with None.
    results_list = [None if not x else x for x in results_list]

    return results_list


def check_column_values(df, col1, col2):
    # Check if either of the two columns contains a non-null value
    result = (df[col1].notnull() | df[col2].notnull()).tolist()
    return result
    
    
    



    

