import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt

#Enable logging for gensim
import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.ERROR)

import warnings
warnings.filterwarnings('ignore',category=DeprecationWarning)

# NLTK stop words
from nltk.corpus import stopwords
stop_words=stopwords.words('english')+['good','app','plant','farmer','useful']


import os

df=pd.read_excel("./dataset/Book2.xlsx" )
df.columns
df=df[['content']]
df=df.dropna()
df['content']=df['content'].astype(str)
df['content']=df['content'].apply(lambda x: x.encode("utf-8").decode("ascii","ignore"))


# Tokenize words and Clean-up text
data= df.content.values.tolist()
def content_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence),deacc=True)) #deacc=True removes punctuations

data_words=list(content_to_words(data))
print(data_words[:1])
count=[len(sublist) for sublist in data_words ]
df['words']=data_words
df['review_len']=count
data_words=[sublist for sublist in data_words if len(sublist)>1]
df=df[df['words'].map(len) >1]

# Creating Bigram and Trigram Models
bigram=gensim.models.Phrases(data_words,min_count=5,threshold=10) #higher threshold fewer phrase
trigram=gensim.models.Phrases(bigram[data_words],threshold=10)

#faster way to get a sentence clubbed as a trigram/bigram
bigram_mod=gensim.models.phrases.Phraser(bigram)
trigram_mod=gensim.models.phrases.Phraser(trigram)


# Remove Stopwords, Make Bigrams and Lemmatize
# Define functions for stopwards, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return[[word for word in simple_preprocess(str(doc))if word not in stop_words]for doc in texts]

def make_bigram(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return[trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN','ADJ','VERB',"ADV"]):
    """https://spacy.io/api/annotation"""
    texts_out=[]
    for content in texts:
        doc=nlp(" ".join(content))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out



# form bigrams
data_words_bigram= make_bigram(data_words)
print(data_words_bigram[:1])

data_words_trigram=make_trigrams(data_words)
print(data_words_trigram[:1])

nlp=spacy.load("en_core_web_sm",disable=['parser,ner'])
data_lemmatized= lemmatization(data_words_trigram,allowed_postags=['NOUN','ADJ','VERB','ADV'])

print(data_lemmatized[:1])
df['words']=data_lemmatized

data_lemmatized=[sublist for sublist in data_lemmatized if len(sublist)>1]
df=df[df['words'].map(len) >1]

id2word= corpora.Dictionary(data_lemmatized)

#Create Corpus
texts=remove_stopwords(data_lemmatized)
df['words']=texts
texts=[sublist for sublist in texts if len(sublist)>1]
df=df[df['words'].map(len) >1]




#Term Document Frequency
corpus=[id2word.doc2bow(text) for text in texts]

print(corpus[:1])

#If you want to see what word a given id corresponds to, pass the id as a key to the dictionary.
id2word[0]

[[(id2word[id],freq) for id, freq in cp]for cp in corpus[:1]]

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,
                                         num_topics=num_topics,random_state=100,
                                         alpha='auto',per_word_topics=True)
        model_list.append(model)

        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=texts, start=2, limit=20, step=2)

#Show graph
limit=20; start=2; step=2;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()



lda_model= gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,
                                           num_topics=4,random_state=100,
                                           alpha='auto',per_word_topics=True)

# Print the Keyword in the 10 topics
doc_lda=lda_model[corpus]
pprint(lda_model.print_topics())


#compute perplexity
lda_perplexity=lda_model.log_perplexity(corpus)
print('\nPerplexity:',lda_perplexity)# a measure of how good the model is, lower the better.


#compute coherence score
coherence_model_lda=CoherenceModel(model=lda_model,texts=texts,dictionary=id2word,coherence='u_mass')
coherence_lda=coherence_model_lda.get_coherence()
print('\nCoherence Score:',coherence_lda)

vis=pyLDAvis.gensim_models.prepare(lda_model,corpus,id2word)
pyLDAvis.save_html(vis,"agri2.html")

# 1. Wordcloud of Top N words in each topic
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  max_words=50,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = lda_model.show_topics(num_words=50,formatted=False)

fig, axes = plt.subplots(2, 2, figsize=(20,20), sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud,interpolation='bilinear')
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=5, hspace=5)
plt.axis('off')
plt.margins(x=5, y=5)
plt.tight_layout()
plt.show()

def format_topics_sentences(ldamodel=None, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
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


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=texts)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.head(10)
df[['Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords']]=df_dominant_topic[['Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords']].values

import  os
file_path = os.getcwd()
file_name = 'agri2 .xlsx'
save_file = os.path.join(file_path, file_name)
df.to_excel(save_file,
                 engine='openpyxl',
                 startrow=0,
                 startcol=0,
                 header=True,
                 na_rep='NaN',
                 float_format='%.2f',
                 sheet_name='Sheet1'
                 )


#pyLDAvis 저장
#pyLDAvis.save_html(vis,"200314_after_neg_lda_model_4.html")

from gensim.test.utils import datapath
#saving model to disk.
temp_file = datapath("agri_model")
lda_model.save(temp_file)


#loading model from disk

from gensim import  models

lda = models.ldamodel.LdaModel.load(temp_file)

#로드
# temp_file = datapath("agri_model")
# agri_model= models.ldamodel.LdaModel.load(temp_file)
# vis = pyLDAvis.gensim_models.prepare(neg_lda4, neg_corpus, neg_id2word)
# pyLDAvis.save_html(vis,"200313_after_neg_lda_model_4.html")

all_topics = {}
num_terms = 10  # Adjust number of words to represent each topic
lambd = 1
# Adjust this accordingly based on tuning above
topic_Term = []
topic_relevance = []
for i in range(1, 5):  # Adjust this to reflect number of topics chosen for final LDA model
    topic = vis.topic_info[vis.topic_info.Category == 'Topic' + str(i)].copy()
    topic['relevance'] = topic['loglift'] * (1 - lambd) + topic['logprob'] * lambd
    topic_Term.append(topic['Term'])
    topic_relevance.append(topic['relevance'])
    all_topics['Topic ' + str(i)] = topic.sort_values(by='relevance', ascending=False).Term[:num_terms].values

pd.DataFrame(all_topics).T


wc = WordCloud(width=1000, height=1000, background_color="white")

plt.figure(figsize=(30,30))
for t in range(lda_model.num_topics):
    plt.subplot(2,2,t+1)
    x = dict(zip(topic_Term[t],topic_relevance[t]))
    im = wc.generate_from_frequencies(x)
    plt.imshow(im)
    plt.axis("off")
    plt.title("Topic #" + str(t+1), size=50)

plt.show()