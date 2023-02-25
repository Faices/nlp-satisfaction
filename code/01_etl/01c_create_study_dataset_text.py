import pandas as pd
from code.functions import *
import timeit

start = timeit.default_timer()
print(start)

## Import dataframe
filelocation = 'data/DataClean'
df = pd.read_feather(filelocation)

###### Stopword list creation #########

# import custom stopwords list
customstopwords = pd.read_excel('config/customstopwords.xlsx')
customstopwords = customstopwords['stopword'].tolist()

# Also add ortsnamen to the stoplist because we have them in the metadata and dont want them in the comments
orte = [x.lower() for x in set(df.ft_startort.tolist()) if x == x and x.lower() != '']

# Create the list of locations
for location in df.ft_startort.tolist():
    # Check if the value is a string
    if isinstance(location, str):
        # Convert to lowercase and remove 'Zug'
        location = location.lower()
        if location == 'zug':
            continue
        
        # Split the location into tokens if it contains whitespace
        tokens = location.split()
        
        # Add each token to the list individually
        for token in tokens:
            # Skip any token that is in the stoplist
            if token in orte:
                continue
            # Remove any commas from the end of the token
            token = token.rstrip(',')
            orte.append(token)
    
# Remove duplicates from the list
orte = list(set(orte))

orte.remove("zug")

# extend the stopword list with the ortsnamen
customstopwords.extend(orte)

## Keep only surveys with filled out "Kommentar"
df_text = df.dropna(subset=["Kommentar"])

#df_text = df_text.head(2000)
print(len(df_text))

df_text = df_text[df_text.Kommentar.apply(lambda x: len(str(x))>=3)] # min 3 characters for valid comment
df_text.reset_index(inplace=True, drop=True)


## Add basic text features
df_text["Kommentar"] = remove_redundant_whitespaces(df_text["Kommentar"]) #note: imported function "remove_redundant_whitespaces"
df_text = add_basic_textfeatures(df_text,"Kommentar")

## Preprocess text
#preprocess_text(df_text, 'Kommentar', locations=didok)
preprocess_text(df_text, 'Kommentar', custom_stopwords=customstopwords)



# print('B')

# ## Tokenize
# from nltk.tokenize import word_tokenize
# df_text['tokenized'] = df_text['Kommentar'].apply(word_tokenize)

# print('C')

# ## Lower
# df_text['lower'] = df_text['tokenized'].apply(lambda x: [word.lower() for word in x])

# print('D')

# ## Removing Punctuations
# punc = string.punctuation
# df_text['no_punc'] = df_text['lower'].apply(lambda x: [word for word in x if word not in punc])

# print('E')

# ## Removing German Stopwords
# stop_words = set(stopwords.words('german'))
# print(stop_words)
# df_text['stopwords_removed'] = df_text['no_punc'].apply(lambda x: [word for word in x if word not in stop_words])

# print('F')

# ## Add string Version of stopwords_removed column to analyse sentiment
# df_text['stopwords_removed_str'] = [' '.join(map(str,l)) for l in df_text['stopwords_removed']]

# print('G')

# ## Removing gemeindenamen
# print(gemeindenamen)
# df_text['locations_removed'] = df_text['stopwords_removed'].apply(lambda x: [word for word in x if word not in gemeindenamen])

# print('H')

# ## Add string Version of locations_removed column to analyse sentiment
# df_text['locations_removed_str'] = [' '.join(map(str,l)) for l in df_text['locations_removed']]

# print('I')


# def lemmatize_words(texts):
#     texts_out = []
#     nlp = spacy.load('de_core_news_lg')
#     for sent in texts:
#         doc = nlp(" ".join(sent))
#         lemmas = []
#         for token in doc:
#             if token.lemma_ == token.text:
#                 lemma = token.lemma_.lower()
#             else:
#                 lemma = token.lemma_
#             lemmas.append(lemma)
#         texts_out.append(lemmas)
#     return texts_out

# # Apply function
# df_text['lemmatized'] = lemmatize_words(df_text['locations_removed'])

# print('J')

# print('J')


# ## Add string Version of lemmatized column to analyse sentiment
# df_text['lemmatized_str'] = [' '.join(map(str,l)) for l in df_text['lemmatized']]

# print(df_text['lemmatized_str'])


# print('K')

# ## PoS Tagging
# def pos_tagging(texts):
#     tags = []
#     for x in texts:
#         blob = TextBlobDE(x)
#         tags.append(blob.tags)
#     return tags

# # Apply function
# df_text['pos_tagged'] = pos_tagging(df_text['lemmatized_str'])

# #Only Nouns

# # create a new column to store the filtered nouns
# df_text['nouns'] = None

# # iterate through each row of the dataframe
# for index, row in df_text.iterrows():
#     # filter out only the nouns using a list comprehension
#     nouns = [word[0] for word in row['pos_tagged'] if word[1] == 'NN']
#     # store the filtered nouns in the new column
#     df_text.at[index, 'nouns'] = nouns

# ## Add string Version of the column
# df_text['nouns_str'] = [' '.join(map(str,l)) for l in df_text['nouns']]


# Add Additional data columns for better slicing
add_date_columns(df_text, 'u_date')

# Sort dataframe by date (newest first)
df_text = df_text.sort_values("u_date",ascending=False)


############## Export ############## 
df_text = df_text.reset_index(drop=True)
df_text.to_feather('data/DataText') # store data in feather file

end = timeit.default_timer()
print("Duration: ",end-start)












