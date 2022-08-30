## NMF
## This is a new file that implements NMF using Spacy on the spelling corrected database (lda_text_newshist_new_cntd_LDA.py runs it on non-spell checked data) and uses a completely new algorithm to extract topics (Refixed December 01,2018).
#Source: https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24
## Source:https://medium.com/ml2vec/topic-modeling-is-an-unsupervised-learning-approach-to-clustering-documents-to-discover-topics-fdfbf30e27df

###########################################################################################################################
###########################################################################################################################

import pickle
import pandas as pd
import pprint
import numpy as np
import sklearn
import nltk
nltk.download('punkt')
import nltk.data
#nltk.download()
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize;
# spacy for lemmatization
import spacy
#import seaborn as sns
import datetime as datetime
#import matplotlib.pyplot as plt
#import matplotlib.dates as mdates
import calendar
import scipy as sp;
import sys;
# Gensim
import gensim
from gensim import corpora
from gensim.models import Phrases
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import CoherenceModel
from gensim.models import Word2Vec
#from ds_voc.text_processing import TextProcessing
from nltk import PorterStemmer
stemmer = PorterStemmer()
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from gensim.models import ldamodel
import gensim.corpora;
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import base64
import re
from gensim import corpora, models
from gensim.models.ldamulticore import LdaMulticore
import itertools
from itertools import chain
import datefinder
from textblob import TextBlob
import os
from itertools import islice
import string
from string import punctuation
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import seaborn as sns
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from operator import attrgetter
from nltk import word_tokenize
import enchant
from spellchecker import SpellChecker
from autocorrect import spell
from sklearn.naive_bayes import MultinomialNB
from symspellpy.symspellpy import SymSpell, Verbosity  # import the module
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
stemmer= PorterStemmer()
import timestring

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

sys.path.insert(0, "/Users/geetagarg/env/lib/python3.7/site-packages/symspellpy")

###########################################################################################################################################################
random_words = ["miss", "mr", "mrs", "said", "jr", "ms", "take", "us", "go", "back","saw", "look", "looks", "already", "would", "should", "see", "also", "could", "new", "yet", "one", "time", "year", "whose", "ever", "may", "couldnt", "shouldnt", "wouldnt", "continued", "in", "st", "roe", "good", "called", "told", "little", "ris", "week", "point", "ago", "mak", "seen", "aug", "wednesday", "goes", "use", "june", "ijn", "bya", "paul", "sit", "sensitive", "lhe", "speaks", "jul", "the", "thus", "offs", "sept", "per", "jun", "pau", "bi", "likely", "fact", "know", "january", "previously", "noted", "partly", "asked", "plus", "currently","tell", "man", "yes", "thing", "school", "sex", "fri", "ewe", "began", "whats", "monday", "story", "request"]
proquest_words = [ "blocked", "due", "copyright", "see", "full", "image", "microfilm", "reproduced", "with", "permission", "copyright", "owner", "further", "reproduction", "prohibited", "without", "proquest", "historical", "newspapers", "chicago", "tribune", "newyork", "york", "times", "pg", "page", "boston", "globe", "calls", "san", "francisco", "san", "percent", "united", "states", "months", "nov", "april", "go", "second", "like", "oct", "isnt", "seven", "later", "set", "come", "home", "view", "yesterday", "telephone", "probably", "rang", "act", "file", "meet", "open", "line", "big", "recent", "check", "washington", "day", "face", "tuesdays", "background", "with", "index", "tuesday", "want", "friday", "eee", "ate", "well", "ad", "year", "got", "eta", "ert", "ole", "tee", "ere", "ee", "cee", "saturday", "came", "wsj", "alee"]
nytimes_words = ["photo", "nytimes", "nytoday", "com", "a", "off", "or", "om", "dr", "ny", "in", "st", "newspapers", "today", "pm", "-", "tions", "jan", "jul", "july",  "aug", "need", "march", "news", "arthur", "wall", "street", "journal", "soon", "meet", "right", "think", "says", "way", "major", "meet", "leonard", "silk", "sep", "dec", "feb", "jan", "march", "apr", "oct", "wall", "street", "journal" ]
STOPLIST = set(stopwords.words('english') + random_words + nytimes_words + proquest_words)
#STOPLIST = set(stopwords.words('english') + proquest_words)
STOPLIST = list(STOPLIST)
#print("stoplist", STOPLIST)

###########################################################################################################################
## Step_1: Read the data

##########################################################################################################################################
## Step_1: Read the data
##########################################################################################################################################

data_text_1 = pickle.load(open('lda_text_newshist_new_spellchecked.dat', 'rb')) ## spellchecked data
data_text_1  = data_text_1.reset_index(drop = True)
print("data_text_1", data_text_1 )
#data_text_1.drop(columns=['dates', 'newpaper'])

#print("data_text_1", data_text_1 )
data_text_dates_names = pd.read_csv('data_text_final.csv', encoding = 'ISO-8859-1')  # Reading the data for dates and newspaper names extracted in the file new_date_newspapername_extraction
data_text_1['dates'] = data_text_dates_names['final_dates']  ## override old dates with new dates
data_text_1['newpaper'] = data_text_dates_names['final_newspaper_names'] ## override old newspaper names with new newspaper names

##########################################################################################################################################
##########################################################################################################################################

data_text_1.drop([795], inplace=True)  ## change this once you remove one more article in the original data (These articles are the major news summaries
data_text_1  = data_text_1.reset_index(drop = True)
print("data_text_before_final", data_text_1)

def extract_month_year(date):
    str_date = str(date)
    # print("str_date", str_date)
    datee = datetime.datetime.strptime(str_date, "%Y-%m-%d")
    month_name = calendar.month_abbr[datee.month]
    year = datee.year
    day = datee.day
    return month_name + " " + str(year)  ## extract only month and year


#data_text_1['Month_Year'] = data_text_1['dates'].apply(lambda x: extract_month_year(x))
#data_text_1['Month_Year']= data_text_1['Month_Year'].apply(lambda x: datetime.datetime.strptime(x, '%b %Y').date()) ## convert month year to numerical dates to be able to sort according to dates. Convert them back to month year once sorted.
#data_text_1['new_Month_Year'].iloc[idx] = datetime.datetime.strptime(data_text_1['Month_Year'].iloc[idx], '%b %Y').date()
#data_text_sorted = data_text_1.sort_values(by = 'Month_Year').reset_index(drop = True)


## manually change these errors
data_text_1.set_value(data_text_1.index[1388], 'dates', 'apr 16 1981')
data_text_1.set_value(data_text_1.index[1388], 'newpaper', 'new york times')
data_text_1.set_value(data_text_1.index[1262], 'dates', 'jan 7 1981')
data_text_1.set_value(data_text_1.index[1262], 'newpaper', 'new york times')


##########################################################################################################################################
## CLEAN THE TEXT
##########################################################################################################################################



print("data_text_final final", data_text_1)
data_text_1.to_csv("final_data_newspapers.csv")

#data_text_1['format_dates'] = "none"
for idx in range(len(data_text_1)):
    data_text_1['Content'].iloc[idx] = ' '.join([word for word in data_text_1['Content'].iloc[idx].split() if word not in STOPLIST]); ## REMOVE STOP WORDS

data_text_1['sort_dates'] = "none"
for idx in range(len(data_text_1)):
    try:
        #data_text_1["sort_dates"].iloc[idx]  = data_text_1["dates"].iloc[idx].apply(lambda x: datetime.datetime.strptime(x, '%b %d %Y').date())
        data_text_1['sort_dates'].iloc[idx]  = datetime.datetime.strptime(data_text_1['dates'].iloc[idx], '%b %d %Y').date()
    except ValueError:
        #  raise ValueError("Incorrect data format, should be YYYY-MM-DD")
        data_text_1['sort_dates'].iloc[idx] =  "dec 11 2020"
        data_text_1['sort_dates'].iloc[idx] = datetime.datetime.strptime(data_text_1['sort_dates'].iloc[idx], '%b %d %Y').date()



print("data_text_final", data_text_1)
pickle.dump(data_text_1 , open('data_text_1_format_dates_dec012018.dat', 'wb'))

##########################################################################################################################################
## Step_1: Read the data
##########################################################################################################################################

#data_text_1 = pickle.load(open('lda_text_newshist_new.dat', 'rb'))
#print("data", data_text_1)


data = data_text_1.Content.values.tolist()
#print("data2", data)


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))

print("data_words", data_words[:1])
#print(data_words)



##########################################################################################################################################
##########################################################################################################################################

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=150) # higher threshold fewer phrases. Use bigram/trigram to extract information in compund words
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print("trigram", trigram_mod[bigram_mod[data_words[0]]])


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in STOPLIST] for doc in texts]

def len_words(texts):
    return [[word for word in simple_preprocess(str(doc)) if len(word) > 2] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


## Be careful in using this. Can also mess up the data
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        #   print("sent", sent)
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

lemmatizer=WordNetLemmatizer()
def preprocess(texts):
    result = []
    return [[lemmatizer.lemmatize(word) for word in gensim.utils.simple_preprocess(str(doc)) if word not in gensim.parsing.preprocessing.STOPWORDS and len(word) > 2] for doc in texts]



# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

data_words_len = len_words(data_words_nostops)

# Form Bigrams
#data_words_bigrams = make_bigrams(data_words_len)

#print("bigrams", data_words_bigrams)

data_stemmed = preprocess(data_words_len)

print("data_stemmed", data_stemmed[:1])


# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
#data_lemmatized = lemmatization(data_stemmed, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

#print("data_lemmatized", data_lemmatized[:1])

pickle.dump(data_stemmed , open('data_lemmetized_dec012018.dat', 'wb'))

print("type", type(data_stemmed))


##########################################################################################################################################
## Get the data ready for NMF
##########################################################################################################################################
#num_topics = 5;

## create the training corpus

train_text= [' '.join(text) for text in data_stemmed] ## This converts a list of lists into a list of strings
print("train_text", train_text[:1])

vectorizer = CountVectorizer(analyzer='word', stop_words=STOPLIST, max_features=10000, min_df = 0.002); #words that are repeated in less than 0.3% documents are dropped

x_counts = vectorizer.fit_transform(train_text);  ## document-term matrix
print("x_counts", pd.DataFrame(x_counts.toarray()))


#Next, we set a TfIdf Transformer, and transform the counts with the model.
transformer = TfidfTransformer(smooth_idf=False); # smooth_idf = false adds “1” to the idf for terms with zero idf, i.e., terms that occur in all documents in a training set, will not be entirely ignored.
x_tfidf = transformer.fit_transform(x_counts);

#And now we normalize the TfIdf values to unit length for each row.
xtfidf_norm = normalize(x_tfidf, norm='l2', axis=1)
print("xtfidf_norm", pd.DataFrame(x_tfidf.toarray()))


#xtfidf_dataframe = pd.DataFrame(xtfidf_norm.max(axis=1),xtfidf_norm.min(axis=1), xtfidf_norm.mean(axis=1))
#print("xtfidf_dataframe", xtfidf_dataframe)

#the word ids obtained need to be reverse-mapped to the words so we can print the topic names.
feat_names = vectorizer.get_feature_names()
#print("feat_names", feat_names)


tf = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word', min_df = 0.002, max_features=10000, stop_words=STOPLIST)
txt_fitted = tf.fit(train_text)
txt_transformed = txt_fitted.transform(train_text)
#print("tf.vocabulary_", tf.vocabulary_)


# get feature names
feature_names = np.array(tf.get_feature_names())
print("length_feature_names", len(feature_names))
new1 = tf.transform(train_text)
print("new1", new1.todense())
# find maximum value for each of the features over all of dataset:
max_val = new1.max(axis=0).toarray().ravel()

#sort weights from smallest to biggest and extract their indices
sort_by_tfidf = max_val.argsort()

print("sort_by_tfidf", sort_by_tfidf)

print("Features with lowest tfidf:\n{}".format(
                                               feature_names[sort_by_tfidf[:500]]))

print("\nFeatures with highest tfidf: \n{}".format(
                                                   feature_names[sort_by_tfidf[-500:]]))






##################################################################################################################################w###################################
## 3)  calculating word frequencies and tfidf using only a few words by creating a disctionary for only few specific words
## This does not fit the NMF model but only calculates the count scores (Dictionary Based method)

## This section also regroups the data and sorts the dataframe according to dates
##################################################################################################################################w###################################

def extract_month_year(date):
    str_date = str(date)
    # print("str_date", str_date)
    datee = datetime.datetime.strptime(str_date, "%Y-%m-%d")
    month_name = calendar.month_abbr[datee.month]
    year = datee.year
    day = datee.day
    return month_name + " " + str(year)  ## extract only month and year



## Vocabulary corresponding to which counts will be calculated
count_vectorizer  = sklearn.feature_extraction.text.CountVectorizer(vocabulary=['volcker', 'appointment', 'job','position','fed', 'federal', 'reserve', 'board', 'chairman', 'chairmanship','conservative', 'short', 'term', 'interest', 'rate', 'rates','fund', 'high','hike', 'rise', 'raise','risen','raising', 'increase','increased','increasing', 'increment','anti', 'inflation', 'inflationary','hyperinflation', 'expectation', 'expectations', 'dollar', 'weak','weakening', 'weakness', 'recession','recessionary','monetary', 'policy','tight', 'money', 'supply', 'devalue', 'devalued','uncertain', 'uncertainty'])

vocabulary=['volcker', 'appointment', 'job','position','fed', 'federal', 'reserve', 'board', 'chairman', 'chairmanship','conservative', 'short', 'term', 'interest', 'rate', 'rates','fund', 'high','hike', 'rise', 'raise','risen','rasing', 'increase','increased','increasing', 'increment','anti', 'inflation', 'inflationary','hyperinflation', 'expectation', 'expectations', 'dollar', 'weak','weakening', 'weakness', 'recession','recessionary','monetary', 'policy','tight', 'money', 'supply', 'devalue', 'devalued','uncertain', 'uncertainty']



freq_term_matrix = count_vectorizer.fit_transform(train_text)
FTerm_tfidf = pd.DataFrame(freq_term_matrix.todense(), columns=vocabulary)
FTerm_tfidf["dates"] = data_text_1['sort_dates'].apply(lambda x: extract_month_year(x))
#FTerm_tfidf = FTerm_tfidf.sort_values(by = 'dates') #.reset_index(drop = True)
print("freq_term_matrix", FTerm_tfidf)
FTerm_tfidf.to_csv('/Users/geetagarg/newshist_nmf_lda/count_articles_alldocs.csv') ## This saves the dataframe in a csv file

data_text_1['Month_Year'] = data_text_1['sort_dates'].apply(lambda x: extract_month_year(x))
data_text_1['Month_Year']= data_text_1['Month_Year'].apply(lambda x: datetime.datetime.strptime(x, '%b %Y').date()) ## convert month year to numerical dates to be able to sort according to dates. Convert them back to month year once sorted.
#data_text_1['new_Month_Year'].iloc[idx] = datetime.datetime.strptime(data_text_1['Month_Year'].iloc[idx], '%b %Y').date()
#iddx = pd.date_range('01-01-2013', '09-30-2013')


data_text_sorted = data_text_1.sort_values(by = 'Month_Year').reset_index(drop = True)

FTerm_tfidf_grouped = FTerm_tfidf.groupby("dates").sum().reset_index() ## Grouping data by sum of keywords and index resetted (articles will now be identified by new index)
print("FTerm_tfidf_grouped", FTerm_tfidf_grouped)




print("data_text_1_sorted", data_text_sorted) ## USE this sorted data to compute aggregated document-topic probabilties
FTerm_tfidf_grouped_no = data_text_1.groupby('Month_Year').size().reset_index() ## grouping data by sum of counts of articles in a month
FTerm_tfidf_grouped_no.columns = ["Month_Year", "Number of Articles"]
FTerm_tfidf_grouped_no = FTerm_tfidf_grouped_no.sort_values(by = 'Month_Year').reset_index(drop = True)
print("FTerm_tfidf_grouped_no", FTerm_tfidf_grouped_no)
FTerm_tfidf_grouped_no_1 = data_text_1[data_text_1['newpaper'] == "wall street journal"]
FTerm_tfidf_grouped_no_1  = FTerm_tfidf_grouped_no_1.groupby('Month_Year').size().reset_index()
print("FTerm_tfidf_grouped_1", FTerm_tfidf_grouped_no_1)
FTerm_tfidf_grouped_no_1.to_csv('article_count_wsj.csv') ## This saves the dataframe in a csv file
FTerm_tfidf_grouped_no_2 = data_text_1[data_text_1['newpaper'] == "new york times"]
FTerm_tfidf_grouped_no_2  = FTerm_tfidf_grouped_no_2.groupby('Month_Year').size().reset_index()
print("FTerm_tfidf_grouped_2", FTerm_tfidf_grouped_no_2)
FTerm_tfidf_grouped_no_2.to_csv('article_count_nytimes.csv') ## This saves the dataframe in a csv file


FTerm_tfidf_grouped["Month_no"] = FTerm_tfidf_grouped["dates"].apply(lambda x: datetime.datetime.strptime(x, '%b %Y').date())
FTerm_tfidf_grouped = FTerm_tfidf_grouped.sort_values(by = "Month_no").reset_index(drop = True)
FTerm_tfidf_grouped["Number of Articles"] = FTerm_tfidf_grouped_no["Number of Articles"]
FTerm_tfidf_grouped_no["orig_dates"] = FTerm_tfidf_grouped["dates"]
print("FTerm_tfidf_grouped_no_again", FTerm_tfidf_grouped_no)
print("FTerm_tfidf_grouped_final",FTerm_tfidf_grouped)


FTerm_tfidf_grouped.to_csv('/Users/geetagarg/newshist_nmf_lda/count_articles_alldocs_grouped.csv') ## This saves the dataframe in a csv file




#######################################################################################################################################################################
## 2) Fitting the NMF model on the whole dataset.
#######################################################################################################################################################################
num_topics = 5  ## Number of topics sought

model =  NMF(n_components=num_topics, random_state=1, max_iter=200, alpha=0.0, l1_ratio=.5, init='nndsvd', tol=1e-4 )
#fit the model
model.fit(xtfidf_norm)  ## Runs NMF model on all documents with document-term matrix contains xtfidf-scores.

#Generating NMF topics:
#We are going to iterate over each topic, obtain the most important scoring words in each cluster, and add them to a Dataframe
def get_nmf_topics(model, n_top_words):
    
    #the word ids obtained need to be reverse-mapped to the words so we can print the topic names.
    feat_names = vectorizer.get_feature_names()
    #print("model.components",pd.DataFrame(model.components_ ))
    # document_topic_probs = pd.DataFrame(model.fit_transform(xtfidf_norm))
    #for
    # print("model.fit_transform",pd.DataFrame(model.fit_transform(xtfidf_norm)))
    #df[len(df.columns)].argmax()
    word_dict = {};
    for i in range(num_topics):
        #for each topic, obtain the largest values, and add the words they map to into the dictionary.
        
        words_ids = model.components_[i].argsort()[:-60- 1:-1]  # Show top 60 words ## topic word distributions (H matrix)
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = words;
        NMF_word_dict = pd.DataFrame(word_dict);
    print("NMF_word_dict_alldata=", NMF_word_dict)
    NMF_word_dict.to_csv("NMF_word_dic.csv")
    NMF_word_dict.to_csv('/Users/geetagarg/newshist_nmf_lda/topics_NMF_newshist_topics=12.csv') ## This saves the dataframe in a csv file
    
    
    n_docs = len(data_text_1.iloc[0:])
    document_indices =[];
    for i in range(n_docs):
        #for each topic, obtain the corresponding documents.
        document_ids = model.fit_transform(xtfidf_norm)[i].argsort()[:-num_topics -1:-1]
        #transformed_data = model.fit_transform(xtfidf_norm)
        document_indices.append(document_ids)
        
        doc_indices = pd.DataFrame(document_indices)
        doc_indices.columns = ['Topic 0', 'Topic 1', 'Topic 2', 'Topic 3', 'Topic 4']
        topic_pr = model.transform(xtfidf_norm)
    print("topic_pr", pd.DataFrame(topic_pr))
    pd.DataFrame(topic_pr).to_csv('NMF_topic_probs.csv')
    print("NMF_document_indicesalldata=", doc_indices)
    doc_indices.to_csv('/Users/geetagarg/newshist_nmf_lda/doc_topics_NMF_newshist_topics=12.csv') ## This saves the documents that have the highest probability of falling under a certain topic
    
    def extract_month_year(date):
        str_date = str(date)
        # print("str_date", str_date)
        datee = datetime.datetime.strptime(str_date, "%Y-%m-%d")
        month_name = calendar.month_abbr[datee.month]
        # month_name = datee.month
        year = datee.year
        day = datee.day
        return  month_name  + " " + str(year) ## extract only month and year
    final_doc_indices = doc_indices[['Topic 0']].copy() ## Create a new dataframe with only first column of doc_indices
    final_doc_indices.columns = ['Dominant_Topic'] ## rename the column name to Dominant_Topic
    final_doc_indices['newspaper'] = data_text_sorted['newpaper']
    final_doc_indices['Month_Year'] = data_text_sorted['Month_Year']
    final_doc_indices['Month_Year'] = final_doc_indices['Month_Year'].apply(lambda x: extract_month_year(x))
    final_doc_indices['Month_Year'] = final_doc_indices['Month_Year'].apply(lambda x: datetime.datetime.strptime(x, '%b %Y')).dt.strftime('%Y-%m')
   
    print("final_doc_indices", final_doc_indices)
    NMF_df_dominant_topic_0 = final_doc_indices.loc[(final_doc_indices['Dominant_Topic'] == 0)]
    #or df_dominant_topic_per_document['Dominant_Topic'] == 3)]
    print("df_dominant_topic_0_before", NMF_df_dominant_topic_0)
    NMF_df_dominant_topic_0 = pd.crosstab(NMF_df_dominant_topic_0.Month_Year,NMF_df_dominant_topic_0.newspaper)

    #iddx = pd.period_range(start='1975-01', end='1981-12', freq='M')
    # idx = pd.period_range(start=min(df_dominant_topic_0.index), end=max(df_dominant_topic_0.index), freq='M')
    #print("idx", idx)
    #df_dominant_topic_0.reindex(idx, fill_value=0)

    #df_dominant_topic_0 = df_dominant_topic_0.groupby('Month_Year').size().reset_index()
    print("df_dominant_topic_0_final", NMF_df_dominant_topic_0)

    NMF_df_dominant_topic_0.to_csv("NMF_df_dominant_topic_0.csv")

## Separate the information from one column into two separate columns for each newspaper and manually create a dataframe for each topic
    NMF_df_dominant_topic_1 = final_doc_indices.loc[(final_doc_indices['Dominant_Topic'] == 1)]
    NMF_df_dominant_topic_1 = pd.crosstab(NMF_df_dominant_topic_1.Month_Year,NMF_df_dominant_topic_1.newspaper)
    print("df_dominant_topic_1_final", NMF_df_dominant_topic_1)

    NMF_df_dominant_topic_1.to_csv("NMF_df_dominant_topic_1.csv")
    
    
    NMF_df_dominant_topic_2 = final_doc_indices.loc[(final_doc_indices['Dominant_Topic'] == 2)]
    NMF_df_dominant_topic_2 = pd.crosstab(NMF_df_dominant_topic_2.Month_Year,NMF_df_dominant_topic_2.newspaper)
    print("df_dominant_topic_2_final", NMF_df_dominant_topic_2)

    NMF_df_dominant_topic_2.to_csv("NMF_df_dominant_topic_2.csv")
    
##########################################################################################################################################

    NMF_df_dominant_topic_3 = final_doc_indices.loc[(final_doc_indices['Dominant_Topic'] == 3)]

    NMF_df_dominant_topic_3 = pd.crosstab(NMF_df_dominant_topic_3.Month_Year,NMF_df_dominant_topic_3.newspaper)
    print("df_dominant_topic_3_final", NMF_df_dominant_topic_3)

    NMF_df_dominant_topic_3.to_csv("NMF_df_dominant_topic_3.csv")



##########################################################################################################################################
    NMF_df_dominant_topic_4 = final_doc_indices.loc[(final_doc_indices['Dominant_Topic'] == 4)]

    NMF_df_dominant_topic_4 = pd.crosstab(NMF_df_dominant_topic_4.Month_Year,NMF_df_dominant_topic_4.newspaper)
    print("df_dominant_topic_4_final", NMF_df_dominant_topic_4)

    NMF_df_dominant_topic_4.to_csv("NMF_df_dominant_topic_4.csv")
    
    
    
    
    
    return pd.DataFrame(word_dict);  ## Function ends here
    print(model.vocabulary_)

#Call the function and obtain the topics:
get_nmf_topics(model, 20)
##########################################################################################################################################

