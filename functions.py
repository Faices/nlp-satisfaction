from pandas_profiling import ProfileReport
from dotenv import load_dotenv
import os
import pyodbc
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
from spacy.lang.de.stop_words import STOP_WORDS
import spacy
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
import pandas as pd
from rake_nltk import Rake


###########################################


def sql_azure_connect():
    # Load .env file
    load_dotenv('config/.env')

    # Import credentials for kuzu Azure DB from .env file
    credentials = {
        'SERVER': os.getenv('SERVER_AZURE', "default"),  
        'DATABASE': os.getenv('DATABASE_AZURE', "default"),
        'USERNAME': os.getenv('USERNAME_AZURE', "default"),
        'PASSWORD': os.getenv('PASSWORD_AZURE', "default"),
        'DRIVER': os.getenv('DRIVER_AZURE', "default") 
    }

    connection_string = f"DRIVER={credentials['DRIVER']};SERVER={credentials['SERVER']};DATABASE={credentials['DATABASE']};UID={credentials['USERNAME']};PWD={credentials['PASSWORD']}"
    cnxn = pyodbc.connect(connection_string)

    # Show available tables
    table_names = [x[2] for x in cnxn.cursor().tables(tableType='TABLE')]
    print("Available tables: ",table_names)

    return cnxn 


###########################################


def filter_dateframe_cols(df,cols:list):
    df = df[[cols]]


###########################################


def add_basic_textfeatures(df, colname: str):
    '''This function takes as input a dataframe and the column name with the text and adds basic textfeatures as new columns. The functions returns a new dataframe'''

    dff = df.copy()

    ## Add character count
    dff[colname + '_' + 'Character'] = df[colname].apply(lambda x: len(x))

    ## Add token count (wordcount)
    dff[colname + '_' + 'Tokens'] = df[colname].apply(lambda x: len(str(x).split()))

    ## Add types count (unique wordcount)
    typecount = df[colname].apply(lambda x: len(set(str(x).split())))
    dff[colname + '_' + 'Types'] = typecount

    ## Add TTR (Type-Token Ratio)
    dff[colname + '_' + 'TTR'] = (typecount / dff[colname + '_' + 'Tokens']) * 100

    return dff


###########################################


def remove_redundant_whitespaces(column):
    '''Removes all additional whitespaces from a list ans returns a new list'''
    
    return [re.sub(r'\s+'," ", x).strip() for x in column]


###########################################

def get_top_n_ngrams(corpus, n=None, ngram_range=(1,1)):
    vec = CountVectorizer(ngram_range=ngram_range)
    # check if corpus is a list of lists and flatten it if so
    if isinstance(corpus[0], list):
        flat_corpus = [item for sublist in corpus for item in sublist]
    else:
        flat_corpus = corpus
    vec.fit(flat_corpus)
    bag_of_words = vec.transform(flat_corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


###########################################


# def get_top_n_ngrams(corpus, n=None, ngram_range=(1,1)):
#     vec = CountVectorizer(ngram_range=ngram_range).fit(corpus)
#     bag_of_words = vec.transform(corpus)
#     sum_words = bag_of_words.sum(axis=0) 
#     words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
#     words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
#     return words_freq[:n]



###########################################


def date_split(df, date_col, train_size=0.7, test_size=0.15, val_size=0.15):
    """
    Splits a DataFrame into train, test, and validation sets based on the date column.
    Each set will contain approximately the same number of samples from each month.
    
    Parameters:
        df: DataFrame to be split
        date_col: column name of the date column
        train_size: proportion of samples to include in the training set (default: 0.7)
        test_size: proportion of samples to include in the test set (default: 0.15)
        val_size: proportion of samples to include in the validation set (default: 0.15)
        
    Returns:
        Tuple of train, test, and validation DataFrames
    """
    df_month = df.groupby(df[date_col].dt.month).apply(lambda x: x.sample(frac=1))
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    val_df = pd.DataFrame()
    for month, group in df_month:
        train_month, val_test_month = train_test_split(group, test_size=(val_size + test_size))
        val_month, test_month = train_test_split(val_test_month, test_size=(val_size/(val_size + test_size)))
        train_df = pd.concat([train_df, train_month])
        test_df = pd.concat([test_df, test_month])
        val_df = pd.concat([val_df, val_month])

    return train_df, test_df, val_df


###########################################


def check_column_values(df, col1, col2):
    # Check if either of the two columns contains a non-null value
    result = (df[col1].notnull() | df[col2].notnull()).tolist()
    return result


########################################### 


def add_date_columns(df, date_col):
    df.insert(loc=2, column = 'year', value=df[date_col].dt.year) #create additional year col for viz
    df.insert(loc=3, column = 'month', value=df[date_col].dt.month) #create additional month col for viz
    df.insert(loc=4, column = 'quarter', value=df[date_col].dt.quarter) #create additional quarter col for viz
    df.insert(loc=5, column = 'yearmonth', value = pd.to_datetime(df[['year', 'month']].assign(DAY=1))) #create additional yearmonth col for viz
    df.insert(loc=6, column = 'yearquarter', value = df['year'].astype(str) + '/' + df['quarter'].astype(str)) #create additional yearquarter col for viz
    df.insert(loc=7, column = 'season', value = df['month'].apply(lambda x: 'spring' if x in [3, 4, 5] else ('summer' if x in [6, 7, 8] else ('autumn' if x in [9, 10, 11] else 'winter'))))
    return df


#######################################
################ RAKE #################
    

import pandas as pd
from rake_nltk import Rake

def calculate_trending_keywords_rake(df, group_column, text_column):
    # Create an empty dictionary to store the keywords and scores for each group
    group_keywords = {}
    
    # Get all unique groups in the dataframe
    groups = df[group_column].unique()
    
    # Loop through each group and extract the keywords
    for group in groups:
        # Get the text data for the current group
        group_text = df[df[group_column] == group][text_column]
        
        # Initialize the RAKE object
        r = Rake()
        
        # Loop through each text in the group and extract the keywords
        for text in group_text:
            r.extract_keywords_from_text(text)
            
            # Add the keywords and their scores to the group_keywords dictionary
            for keyword, score in r.get_word_degrees().items():
                if keyword in group_keywords:
                    group_keywords[keyword][group] = score
                else:
                    group_keywords[keyword] = {group: score}
    
    # Create an empty dataframe to store the final results
    results = pd.DataFrame()
    
    # Loop through each group and get the top 10 keywords with the highest scores
    for group in groups:
        # Sort the keywords by their scores
        sorted_keywords = sorted(group_keywords.items(), key=lambda x: x[1].get(group, 0), reverse=True)
        
        # Get the top 10 keywords
        top_10 = sorted_keywords[:10]
        
        # Create a new dataframe with the top 10 keywords and their scores for the current group
        group_df = pd.DataFrame(top_10, columns=['Keyword', 'Score'])
        group_df['Group'] = group
        
        # Add a new column for the rank of each keyword
        group_df['Rank'] = range(1, 11)
        
        # Append the group dataframe to the final results dataframe
        results = results.append(group_df)
    
    return results

#######################################
################ TFIDF #################

# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from collections import Counter

# def calculate_trending_keywords_tfidf(df, group_column, text_column, min_df=0.0, max_df=1):
#     # Create an empty dataframe to store the final results
#     results = pd.DataFrame(columns=['Keyword', 'Score', 'Group', 'Rank','Count','Count Relative'])
#     # Get all unique groups in the dataframe
#     groups = df[group_column].unique()
#     # Initialize the TfidfVectorizer object
#     tfidf = TfidfVectorizer(min_df=min_df, max_df=max_df)

#     for group in groups:
#         # Get the text data for the current group
#         group_text = df[df[group_column] == group][text_column]
#         group_text = group_text.str.cat(sep=' ')
#         # Extract the keywords and their scores
#         tfidf_matrix = tfidf.fit_transform([group_text])
#         feature_names = tfidf.get_feature_names_out()
#         scores = tfidf_matrix.toarray()[0]
#         feature_indexes = scores.argsort()[-10:][::-1]
#         feature_names = [feature_names[i] for i in feature_indexes]
#         feature_scores = [scores[i] for i in feature_indexes]
#         # Create a new dataframe with the top 10 keywords and their scores for the current group
#         group_df = pd.DataFrame({'Keyword':feature_names,'Score':feature_scores})
#         group_df['Group'] = group
#         # Add a new column for the rank of each keyword
#         group_df['Rank'] = range(1, 11)
        
#         group_df['Count'] = [tfidf_matrix[0,tfidf.vocabulary_[word]].round(3) for word in feature_names]
#         group_df["Count Relative"] = group_df["Count"]/group_df["Count"].sum()
        
#         # Append the group dataframe to the final results dataframe
#         results = results.append(group_df)
#     return results



# def convert_to_wide_format(df):
#     # Create an empty dataframe with the groups as the index
#     wide_format = pd.DataFrame(index=df['Group'].unique())
    
#     # Loop through the top 10 keywords and add them as columns to the wide format dataframe
#     for i in range(1, 11):
#         keyword_col = 'Keyword ' + str(i)
#         wide_format[keyword_col] = df[df['Rank'] == i].set_index('Group')['Keyword']
        
#     return wide_format


##########################
######### TFIDF ##########

from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

def find_trending_keywords(dataframe, filter_column, text_column, ngram_range=(1, 1), n=10, min_df=100, max_df=0.2):

    # convert values in filter column to categorical values
    dataframe[filter_column] = dataframe[filter_column].astype('category')

    # add "unknown" category to filter_column categories, if not already present
    if "unknown" not in dataframe[filter_column].cat.categories:
        dataframe[filter_column] = dataframe[filter_column].cat.add_categories("unknown")

    # replace NaN values in filter_column with "unknown"
    dataframe[filter_column].fillna("unknown", inplace=True)

    # Create an empty dictionary to store the top keywords and their counts for each value in filter_column
    trending_keywords = {}

    # Get all values in filter_column
    values = dataframe[filter_column].unique()

    # Convert the tokenized text column to a list of space-separated strings
    text_data = [' '.join(words) for words in dataframe[text_column]]

    # Create a TfidfVectorizer object with the specified n-gram range and min_df parameter
    tfidf_vect = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df)

    # Fit and transform the tokenized text column for the whole corpus
    tfidf = tfidf_vect.fit_transform(text_data)

    # Loop over the values
    for value in values:
        # Filter the dataframe for the given value in filter_column
        filter_data = dataframe[dataframe[filter_column] == value]

        # Convert the tokenized text column to a list of space-separated strings
        text_data_filter = [' '.join(words) for words in filter_data[text_column]]

        # Transform the tokenized text column for the given value using the fitted TfidfVectorizer
        tfidf_filter = tfidf_vect.transform(text_data_filter)

        # Compute the sum of TF-IDF scores for each term in the given value
        tfidf_filter = tfidf_filter.sum(axis=0)

        # Create a list of tuples with the term and its TF-IDF score for the group
        keywords = [(term, tfidf_filter[0, index]) for term, index in tfidf_vect.vocabulary_.items()]

        # Filter out terms that have zero TF-IDF scores
        keywords = [kw for kw in keywords if kw[1] > 0]

        # Sort the keywords based on their TF-IDF scores
        keywords.sort(key=lambda x: x[1], reverse=True)

        # Count the occurrence of each keyword in the group
        group_text_data = ' '.join(text_data_filter)
        group_word_count = Counter(group_text_data.split())

        # Create a list of tuples with the term, its TF-IDF score, and count in the group
        keywords_with_count = [(kw[0], kw[1], group_word_count[kw[0]], group_word_count[kw[0]]/len(group_word_count)) for kw in keywords]

        # Store the top n keywords for the given value in the dictionary
        trending_keywords[value] = keywords_with_count[:n]

    # Return the dictionary of top keywords and their counts for each value in filter_column
    return trending_keywords



##### Compare comments by categorie to see saisonal and temporal trends ###

def find_trending_keywords_diff_normaized(dataframe, filter_column, text_column, ngram_range=(1, 1), n=10, min_df=100, max_df=0.2):

    # convert values in filter column to categorical values
    dataframe[filter_column] = dataframe[filter_column].astype('category')

    # add "unknown" category to filter_column categories, if not already present
    if "unknown" not in dataframe[filter_column].cat.categories:
        dataframe[filter_column] = dataframe[filter_column].cat.add_categories("unknown")

    # replace NaN values in filter_column with "unknown"
    dataframe[filter_column].fillna("unknown", inplace=True)

    # create an empty dictionary to store the top keywords for each value in filter_column
    trending_keywords = {}

    # get all values in filter_column
    values = dataframe[filter_column].unique()

    # convert the tokenized text column to a list of space-separated strings
    text_data = [' '.join(words) for words in dataframe[text_column]]

    # create a TfidfVectorizer object with the specified n-gram range and min_df parameter
    tfidf_vect = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df)

    # fit and transform the tokenized text column for the whole corpus
    tfidf = tfidf_vect.fit_transform(text_data)

    # loop over the values
    for value in values:
        # filter the dataframe for the given value in filter_column
        filter_data = dataframe[dataframe[filter_column] == value]

        # convert the tokenized text column to a list of space-separated strings
        text_data_filter = [' '.join(words) for words in filter_data[text_column]]

        # transform the tokenized text column for the given value using the fitted TfidfVectorizer
        tfidf_filter = tfidf_vect.transform(text_data_filter)

        # compute the sum of TF-IDF scores for each term in the given value
        tfidf_filter = tfidf_filter.sum(axis=0)

        # normalize the TF-IDF scores by the total count of all words in the group
        group_word_count = Counter(' '.join(text_data_filter).split())
        total_count = sum(group_word_count.values())
        tfidf_filter = tfidf_filter / total_count

        # Compute the sum of TF-IDF scores for each term in the other values
        tfidf_other_sum = 0
        for other_value in values:
            if other_value != value:
                # Filter the dataframe for the other value in filter_column
                other_data = dataframe[dataframe[filter_column] == other_value]

                # Convert the tokenized text column to a list of space-separated strings
                text_data_other = [' '.join(words) for words in other_data[text_column]]

                # Transform the tokenized text column for the other value using the fitted TfidfVectorizer
                tfidf_other = tfidf_vect.transform(text_data_other)

                # Compute the sum of TF-IDF scores for each term in the other value
                tfidf_other = tfidf_other.sum(axis=0)

                # normalize the TF-IDF scores by the total count
                total_count = tfidf_other.sum()
                tfidf_other = tfidf_other / total_count

                # Add the normalized TF-IDF scores to the running sum
                tfidf_other_sum += tfidf_other

        # Compute the average of the other values' TF-IDF scores for each term
        tfidf_other_avg = tfidf_other_sum / (len(values) - 1)

        # Compute the difference in TF-IDF scores between the given value and the average of the other values
        tfidf_diff = tfidf_filter - tfidf_other_avg

        # Create a list of tuples with the term and its TF-IDF score difference
        keywords = [(term, tfidf_diff[0, index]) for term, index in tfidf_vect.vocabulary_.items()]

        # Filter out terms that have negative or zero TF-IDF score differences
        #keywords = [kw for kw in keywords if kw[1] > 0]

        # Sort the keywords based on their TF-IDF score difference
        keywords.sort(key=lambda x: x[1], reverse=True)

        # Count the occurrence of each keyword in the group
        group_text_data = ' '.join(text_data_filter)
        group_word_count = Counter(group_text_data.split())

        # Compute the total count of all words in the group
        total_count = sum(group_word_count.values())

        # Create a list of tuples with the term, its TF-IDF score difference, count in the group, and relative count
        keywords_with_count_rel = [(kw[0], kw[1], group_word_count[kw[0]], group_word_count[kw[0]] / total_count) for kw in keywords]

        # Store the top n keywords for the given value in the dictionary with relative count
        trending_keywords[value] = keywords_with_count_rel[:n]

    # Return the dictionary of top keywords for each value in filter_column
    return trending_keywords


    ##########################


# import spacy
# import re
# from spacy.lang.de.stop_words import STOP_WORDS

# def preprocess_text(df, text_column, custom_stopwords=None):
#     nlp = spacy.load("de_core_news_lg")
#     words_to_remove = set(STOP_WORDS) | set(custom_stopwords) if custom_stopwords else set(STOP_WORDS)
    
#     # Lowercase the text, remove stop words and custom stopwords, and remove numbers and special characters
#     text_preprocessed = df[text_column].str.lower().apply(
#         lambda x: " ".join([re.sub(r'[^\w\s]', '', word) for word in re.sub(r'([a-zA-Z]+)-([a-zA-Z]+)', r'\1 \2', x).split() if word not in words_to_remove and not re.search(r'\d', word)])
#     )
    
#     lemmatized = []
#     nouns = []
#     adjectives = []
#     verbs = []
#     nouns_adjectives_and_verbs = []
#     for text in text_preprocessed:
#         doc = nlp(text)
#         if not doc:
#             lemmatized.append([])
#             nouns.append([])
#             adjectives.append([])
#             verbs.append([])
#             nouns_adjectives_and_verbs.append([])
#             continue
        
#         lemmatized_text = []
#         nouns_text = []
#         adjectives_text = []
#         verbs_text = []
#         nouns_adjectives_and_verbs_text = []
#         for token in doc:
#             if not token.text or not token.lemma_:
#                 continue
#             lemmatized_text.append(token.lemma_.lower())
#             if token.pos_ == "NOUN":
#                 nouns_text.append(token.lemma_.lower())
#                 nouns_adjectives_and_verbs_text.append(token.lemma_.lower())
#             if token.pos_ == "ADJ":
#                 adjectives_text.append(token.lemma_.lower())
#                 nouns_adjectives_and_verbs_text.append(token.lemma_.lower())
#             if token.pos_ == "VERB":
#                 verbs_text.append(token.lemma_.lower())
#                 nouns_adjectives_and_verbs_text.append(token.lemma_.lower())
                
#         lemmatized.append(lemmatized_text)
#         nouns.append(nouns_text)
#         adjectives.append(adjectives_text)
#         verbs.append(verbs_text)
#         nouns_adjectives_and_verbs.append(nouns_adjectives_and_verbs_text)
        
#     df["text_preprocessed"] = text_preprocessed
#     df["lemmatized"] = lemmatized
#     df["nouns"] = nouns
#     df["adjectives"] = adjectives
#     df["verbs"] = verbs
#     df["nouns_adjectives_and_verbs"] = nouns_adjectives_and_verbs

#     return df



# import spacy
# import re
# from spacy.lang.de.stop_words import STOP_WORDS

# def preprocess_text(df, text_column, custom_stopwords=None):
#     nlp = spacy.load("de_core_news_lg")
#     words_to_remove = set(STOP_WORDS) | set(custom_stopwords) if custom_stopwords else set(STOP_WORDS)
    
#     # Lowercase the text, remove stop words and custom stopwords, and remove numbers and special characters
#     text_preprocessed = df[text_column].str.lower().apply(
#         lambda x: " ".join([re.sub(r'[^\w\s]', '', word) for word in re.sub(r'([a-zA-Z]+)-([a-zA-Z]+)', r'\1 \2', x).split() if word not in words_to_remove and not re.search(r'\d', word)])
#     )
    
#     lemmatized = []
#     nouns = []
#     adjectives = []
#     verbs = []
#     nouns_adjectives_and_verbs = []
#     text_preprocessed_tokenized = []
    
#     for text in text_preprocessed:
#         doc = nlp(text)
#         if not doc:
#             lemmatized.append([])
#             nouns.append([])
#             adjectives.append([])
#             verbs.append([])
#             nouns_adjectives_and_verbs.append([])
#             text_preprocessed_tokenized.append([])
#             continue
        
#         lemmatized_text = []
#         nouns_text = []
#         adjectives_text = []
#         verbs_text = []
#         nouns_adjectives_and_verbs_text = []
#         text_preprocessed_tokenized_text = []
        
#         for token in doc:
#             if not token.text or not token.lemma_:
#                 continue
#             lemmatized_text.append(token.lemma_.lower())
#             if token.pos_ == "NOUN":
#                 nouns_text.append(token.lemma_.lower())
#                 nouns_adjectives_and_verbs_text.append(token.lemma_.lower())
#             if token.pos_ == "ADJ":
#                 adjectives_text.append(token.lemma_.lower())
#                 nouns_adjectives_and_verbs_text.append(token.lemma_.lower())
#             if token.pos_ == "VERB":
#                 verbs_text.append(token.lemma_.lower())
#                 nouns_adjectives_and_verbs_text.append(token.lemma_.lower())
#             text_preprocessed_tokenized_text.append(token.lemma_.lower())
                
#         lemmatized.append(lemmatized_text)
#         nouns.append(nouns_text)
#         adjectives.append(adjectives_text)
#         verbs.append(verbs_text)
#         nouns_adjectives_and_verbs.append(nouns_adjectives_and_verbs_text)
#         text_preprocessed_tokenized.append(text_preprocessed_tokenized_text)
        
#     df["text_preprocessed"] = text_preprocessed
#     df["text_preprocessed_tokenized"] = text_preprocessed_tokenized
#     df["lemmatized"] = lemmatized
#     df["nouns"] = nouns
#     df["adjectives"] = adjectives
#     df["verbs"] = verbs
#     df["nouns_adjectives_and_verbs"] = nouns_adjectives_and_verbs

#     return df


# import spacy
# import re
# from spacy.lang.de.stop_words import STOP_WORDS

def preprocess_text(df, text_column, custom_stopwords=None):
    nlp = spacy.load("de_core_news_lg")
    words_to_remove = set(STOP_WORDS) | set(custom_stopwords) if custom_stopwords else set(STOP_WORDS)

    # Lowercase the text, remove stop words and custom stopwords, and remove numbers and special characters
    text_preprocessed = df[text_column].str.lower().apply(
        lambda x: " ".join([re.sub(r'[^\w\s]', '', word) for word in re.sub(r'([a-zA-Z]+)-([a-zA-Z]+)', r'\1 \2', x).split() if word not in words_to_remove and not re.search(r'\d', word)])
    )

    tokenized = []
    nouns = []
    adjectives = []
    verbs = []
    nouns_adjectives_and_verbs = []

    for text in text_preprocessed:
        doc = nlp(text)
        if not doc:
            tokenized.append([])
            nouns.append([])
            adjectives.append([])
            verbs.append([])
            nouns_adjectives_and_verbs.append([])
            continue

        tokenized_text = []
        nouns_text = []
        adjectives_text = []
        verbs_text = []
        nouns_adjectives_and_verbs_text = []

        for token in doc:
            if not token.text:
                continue
            token_text = token.text.lower()
            if token_text not in words_to_remove:
                tokenized_text.append(token_text)
                if token.pos_ == "NOUN":
                    nouns_text.append(token_text)
                    nouns_adjectives_and_verbs_text.append(token_text)
                if token.pos_ == "ADJ":
                    adjectives_text.append(token_text)
                    nouns_adjectives_and_verbs_text.append(token_text)
                if token.pos_ == "VERB":
                    verbs_text.append(token_text)
                    nouns_adjectives_and_verbs_text.append(token_text)

        tokenized.append(tokenized_text)
        nouns.append(nouns_text)
        adjectives.append(adjectives_text)
        verbs.append(verbs_text)
        nouns_adjectives_and_verbs.append(nouns_adjectives_and_verbs_text)

    df["text_preprocessed"] = text_preprocessed
    df["text_preprocessed_tokenized"] = tokenized
    df["lemmatized"] = None
    df["nouns"] = nouns
    df["adjectives"] = adjectives
    df["verbs"] = verbs
    df["nouns_adjectives_and_verbs"] = nouns_adjectives_and_verbs

    return df


#####################

def join_list_of_list(list_of_list):
    """
    This function takes in a list of lists and returns a list of strings where each string is made by joining the elements of the corresponding list.
        
    Parameters:
        - list_of_list(List[List[Any]]): List of lists whose elements to be joined
            
    Returns:
         List[str]: List of strings where each string is made by joining the elements of the corresponding list.
    """
    return [' '.join(map(str,l)) for l in list_of_list]



def reduce_dataframe(df, group_column, filter_value):
    """
    Reduces a Pandas dataframe based on a specific column and value.
    
    Parameters:
    df (Pandas dataframe): The dataframe to reduce.
    group_column (str): The name of the column to group the dataframe by.
    filter_value: The value to filter the dataframe on.
    
    Returns:
    A reduced Pandas dataframe.
    """
    
    # Group the dataframe by the specified column
    grouped = df.groupby(group_column)
    
    # Filter the groups based on the filter value
    filtered_groups = {group: data for group, data in grouped if filter_value in data[group_column].values}
    
    # Combine the filtered groups into a new dataframe
    reduced_df = pd.concat(filtered_groups.values())
    
    return reduced_df


###########################


def generate_profiling_report(data_file="DataText", folder_path="data/", report_title=None, report_file="html/ProfilingDataText.html", lazy=False, dark_mode=False, minimal=True):
    """
    Generates a pandas profiling report for the given data file.
    
    Parameters:
    - data_file (str): The name of the data file to be used for generating the report. Default is "DataText".
    - folder_path (str): The path of the folder where the data file is located. Default is "data/".
    - report_title (str): The title to be used for the report. Default is None.
    - report_file (str): The filepath and name of the report file. Default is "html/ProfilingDataText.html"
    - lazy (bool): Whether to load the data in a lazy or non-lazy way. Default is False
    - dark_mode (bool): Whether to use the dark mode or not. Default is False
    - minimal (bool): Whether to produce a minimal report or not. Default is True
    
    Returns:
    None
    """
    # import data
    df = pd.read_feather(folder_path + data_file)
    if report_title is None:
        report_title = data_file
    # Pandas Profiling TextData
    profile = ProfileReport(
        df,
        title=report_title,
        lazy=lazy,
        dark_mode=dark_mode,
        minimal=minimal,
    )
    profile.to_file(report_file)