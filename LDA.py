## This file runs the LDA model on the preprocessed (and spellchecked) text.

## Source: Follow the source code below for model checking - Coherence values and perplexity scores and Pyldavis graphs. Do not use Mallet
## Source: https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
## Source: https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24
## Source: https://medium.com/ml2vec/topic-modeling-is-an-unsupervised-learning-approach-to-clustering-documents-to-discover-topics-fdfbf30e27df (Don't use this for LDA) Use the spacy LDA not gensim LDA
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

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

sys.path.insert(0, "/Users/geetagarg/env/lib/python3.7/site-packages/symspellpy")

###########################################################################################################################################################
random_words = ["miss", "mr", "mrs", "said", "jr", "ms", "take", "us", "go", "back","saw", "look", "looks", "already", "would", "should", "see", "also", "could", "new", "yet", "one", "time", "year", "whose", "ever", "may", "couldnt", "shouldnt", "wouldnt", "continued", "in", "st", "roe", "good", "called", "told", "little", "ris", "week", "point", "ago", "mak", "seen", "aug", "wednesday", "goes", "use", "june", "ijn", "bya", "paul", "sit", "sensitive", "lhe", "speaks", "jul", "the"]
proquest_words = [ "blocked", "due", "copyright", "see", "full", "image", "microfilm", "reproduced", "with", "permission", "copyright", "owner", "further", "reproduction", "prohibited", "without", "proquest", "historical", "newspapers", "chicago", "tribune", "newyork", "york", "times", "pg", "page", "boston", "globe", "calls", "san", "francisco", "san", "percent", "united", "states", "months", "nov", "april", "go", "second", "like", "oct", "isnt", "seven", "later", "set", "come", "home", "view", "yesterday", "telephone", "probably", "rang", "act", "file", "meet", "open", "line", "big", "recent", "check", "washington", "day", "face", "tuesdays", "background", "with", "index"]
nytimes_words = ["photo", "nytimes", "nytoday", "com", "a", "off", "or", "om", "dr", "ny", "in", "st", "newspapers", "today", "pm", "-", "tions", "jan", "jul", "july",  "aug", "need", "march", "news", "arthur", "wall", "street", "journal", "soon", "meet", "right", "think", "says", "way", "major", "meet", "leonard", "silk", "needham", "asked", "moscow", "smith", "bagley", "current", "file", "sep", "dec", "feb", "jan", "march", "apr", "oct", "wall", "street", "journal" ]
STOPLIST = set(stopwords.words('english') + random_words + nytimes_words + proquest_words)
#STOPLIST = set(stopwords.words('english') + proquest_words)
STOPLIST = list(STOPLIST)
print("stoplist", STOPLIST)

##########################################################################################################################################
## Step_1: Read the data
##########################################################################################################################################

data_lemmatized = pickle.load(open('data_lemmetized_dec012018.dat', 'rb'))

##########################################################################################################################################
# Create the Dictionary and Corpus needed for Topic Modeling
##########################################################################################################################################

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])


##########################################################################################################################################
# Build LDA model
##########################################################################################################################################

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=26,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)


# Print the Keyword in the 10 topics
top_words_per_topic = []
for t in range(lda_model.num_topics):
    top_words_per_topic.extend([(t, ) + x for x in lda_model.show_topic(t, topn = 50)])

pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'P']).to_csv("top_words.csv")


doc_lda = lda_model[corpus]


##########################################################################################################################################
# Compute Model Perplexity and Coherence Score
##########################################################################################################################################

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

##########################################################################################################################################
# Visualize the topics-keywords
##########################################################################################################################################

# Visualize the topics
#pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis
pyLDAvis.save_html(vis, 'LDA_Visualization.html')




##########################################################################################################################################
#find the optimal number of topics for LDA? Uncomment this to do model checking
##########################################################################################################################################

#def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
#    """
#        Compute c_v coherence for various number of topics

#Parameters:
#----------
#dictionary : Gensim dictionary
#corpus : Gensim corpus
#texts : List of input texts
#limit : Max num of topics

#Returns:
# -------
#model_list : List of LDA topic models
#        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
#       """
#coherence_values = []
#model_list = []
#for num_topics in range(start, limit, step):
#model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=100,update_every=1, chunksize=100, passes=10,
#                                                alpha='auto', per_word_topics=True)
#        model_list.append(model)
#coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
#   coherence_values.append(coherencemodel.get_coherence())

#   return model_list, coherence_values




# Can take a long time to run.
#model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=40, step=6)


# Show graph
#limit=40; start=2; step=6;
#x = range(start, limit, step)
#plt.plot(x, coherence_values)
#plt.xlabel("Num Topics")
#plt.ylabel("Coherence score")
#plt.legend(("coherence_values"), loc='best')
#plt.show()



# Print the coherence scores
#for m, cv in zip(x, coherence_values):
#    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

## This gives num_topics = 26 at which the coherence score is maximum and after which it starts declining. So for the above analysis, we'll use num_topics = 26.

##########################################################################################################################################
#Finding the dominant topic in each sentence
##########################################################################################################################################

def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_lemmatized):
    # Init output
    sent_topics_df = pd.DataFrame()
    
    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: x[1], reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

# Add original text to the end of the output
contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_lemmatized)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.head(30)


##########################################################################################################################################
# Find the most representative document for each topic
##########################################################################################################################################

# Group top 5 sentences under each topic
sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)],
                                            axis=0)

# Reset Index
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

# Show
sent_topics_sorteddf_mallet.head()



##########################################################################################################################################
# Topic distribution across documents
##########################################################################################################################################

# Number of Documents for Each Topic
topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

# Percentage of Documents for Each Topic
topic_contribution = round(topic_counts/topic_counts.sum(), 4)

# Topic Number and Keywords
topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

# Concatenate Column wise
df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# Change Column names
df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

# Show
df_dominant_topics

