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
stop_words=stopwords.words('english')

import os

df=pd.read_excel("./dataset/netflix.xlsx" )
df.columns
df=df[['content','score','thumbsUpCount']]
df=df[df['score']==5]
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

#see trigram example
print(trigram_mod[bigram_mod[data_words[0]]])

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

# remove stopwords
data_words_nostops= remove_stopwords(data_words)

# form bigrams
data_words_bigram= make_bigram(data_words_nostops)
print(data_words_bigram[:1])

data_words_trigram=make_trigrams(data_words_nostops)
print(data_words_trigram[:1])

nlp=spacy.load("en_core_web_sm",disable=['parser,ner'])
data_lemmatized= lemmatization(data_words_trigram,allowed_postags=['NOUN','ADJ','VERB','ADV'])

print(data_lemmatized[:1])
df['words']=data_lemmatized
data_lemmatized=[sublist for sublist in data_lemmatized if len(sublist)>1]
df=df[df['words'].map(len) >1]

words_bigram=[[item for item in sublist if '_' in item ]for sublist in data_lemmatized]
df['words']=words_bigram
words_bigram=[sublist for sublist in words_bigram if len(sublist)>0]
df=df[df['words'].map(len) >0]
id2word= corpora.Dictionary(words_bigram)

#Create Corpus
texts=(words_bigram)

#Term Document Frequency
corpus=[id2word.doc2bow(text) for text in texts]

#Term Document Frequency
corpus=[id2word.doc2bow(text) for text in texts]

print(corpus[:1])

#If you want to see what word a given id corresponds to, pass the id as a key to the dictionary.
id2word[0]

[[(id2word[id],freq) for id, freq in cp]for cp in corpus[:1]]



lda_model= gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,
                                           num_topics=3,random_state=100,update_every=1,
                                           chunksize=20000, passes=1,iterations=50,
                                           alpha='auto',per_word_topics=True)

# Print the Keyword in the 10 topics
doc_lda=lda_model[corpus]
pprint(lda_model.print_topics())
pprint(lda_model.show_topics())


#compute perplexity
lda_perplexity=lda_model.log_perplexity(corpus)
print('\nPerplexity:',lda_perplexity)# a measure of how good the model is, lower the better.


#compute coherence score
coherence_model_lda=CoherenceModel(model=lda_model,texts=texts,dictionary=id2word,coherence='u_mass')
coherence_lda=coherence_model_lda.get_coherence()
print('\nCoherence Score:',coherence_lda)

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

df1=pd.read_excel("./dataset/netflix.xlsx" )
df1.columns
df1=df1[['content','score','thumbsUpCount']]
df1=df1[df1['score']==1]
df1=df1.dropna()
df1['content']=df1['content'].astype(str)
df1['content']=df1['content'].apply(lambda x: x.encode("utf-8").decode("ascii","ignore"))

# Tokenize words and Clean-up text
data1= df1.content.values.tolist()


data_words1=list(content_to_words(data1))
print(data_words1[:1])
count1=[len(sublist) for sublist in data_words1 ]
df1['words']=data_words1
df1['review_len']=count1
data_words1=[sublist for sublist in data_words1 if len(sublist)>1]
df1=df1[df1['words'].map(len) >1]

# Creating Bigram and Trigram Models
bigram1=gensim.models.Phrases(data_words1,min_count=5,threshold=10) #higher threshold fewer phrase
trigram1=gensim.models.Phrases(bigram[data_words1],threshold=10)

#faster way to get a sentence clubbed as a trigram/bigram
bigram_mod1=gensim.models.phrases.Phraser(bigram1)
trigram_mod1=gensim.models.phrases.Phraser(trigram1)

#see trigram example
print(trigram_mod1[bigram_mod1[data_words1[0]]])


# remove stopwords
data_words_nostops1= remove_stopwords(data_words1)

# form bigrams
data_words_bigram1= make_bigram(data_words_nostops1)
print(data_words_bigram1[:1])

data_words_trigram1=make_trigrams(data_words_nostops1)
print(data_words_trigram1[:1])

nlp=spacy.load("en_core_web_sm",disable=['parser,ner'])
data_lemmatized1= lemmatization(data_words_trigram1,allowed_postags=['NOUN','ADJ','VERB','ADV'])

print(data_lemmatized1[:1])
df1['words']=data_lemmatized1
data_lemmatized1=[sublist for sublist in data_lemmatized1 if len(sublist)>1]
df1=df1[df1['words'].map(len) >1]

words_bigram1=[[item for item in sublist if '_' in item ]for sublist in data_lemmatized1]
df1['words']=words_bigram1
words_bigram1=[sublist for sublist in words_bigram1 if len(sublist)>0]
df1=df1[df1['words'].map(len) >0]
id2word1= corpora.Dictionary(words_bigram1)

#Create Corpus
texts1=(words_bigram1)

#Term Document Frequency
corpus1=[id2word1.doc2bow(text) for text in texts1]


lda_model1= gensim.models.ldamodel.LdaModel(corpus=corpus1,id2word=id2word1,
                                           num_topics=3,random_state=100,update_every=1,
                                           chunksize=20000, passes=1,iterations=50,
                                           alpha='auto',per_word_topics=True)

# Print the Keyword in the 10 topics
doc_lda1=lda_model1[corpus1]
pprint(lda_model1.print_topics())
pprint(lda_model1.show_topics())


#compute perplexity
lda_perplexity1=lda_model1.log_perplexity(corpus1)
print('\nPerplexity:',lda_perplexity1)# a measure of how good the model is, lower the better.


#compute coherence score
coherence_model1_lda=CoherenceModel(model=lda_model1,texts=texts1,dictionary=id2word1,coherence='u_mass')
coherence_lda1=coherence_model1_lda.get_coherence()
print('\nCoherence Score:',coherence_lda1)


df_topic_sents_keywords1 = format_topics_sentences(ldamodel=lda_model1, corpus=corpus1, texts=texts1)

# Format
df_dominant_topic1 = df_topic_sents_keywords1.reset_index()
df_dominant_topic1.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic1.head(10)
df1[['Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords']]=df_dominant_topic1[['Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords']].values
#df1.loc[df1["Dominant_Topic"]==0,'Dominant_Topic']=3
#df1.loc[df1["Dominant_Topic"]==1,'Dominant_Topic']=4
#df1.loc[df1["Dominant_Topic"]==2,'Dominant_Topic']=5
#df1.Dominant_Topic.value_counts()
df.loc[df["Dominant_Topic"]==0,'Dominant_Topic']=3
df.loc[df["Dominant_Topic"]==1,'Dominant_Topic']=4
df.loc[df["Dominant_Topic"]==2,'Dominant_Topic']=5
df.Dominant_Topic.value_counts()

frames=[df,df1]
all=pd.concat(frames)

# 1. Wordcloud of Top N words in each topic
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=30,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = lda_model1.show_topics(num_words=30,formatted=False)+ lda_model.show_topics(num_words=30,formatted=False)


fig, axes = plt.subplots(2, 3, figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()

vis=pyLDAvis.gensim_models.prepare(lda_model,corpus,id2word)
pyLDAvis.save_html(vis,"LDA_visualization bigram score 5.html")
vis=pyLDAvis.gensim_models.prepare(lda_model1,corpus1,id2word1)
pyLDAvis.save_html(vis,"LDA_visualization bigram score 1.html")

file_path = os.getcwd()
file_name = 'bigram .xlsx'
save_file = os.path.join(file_path, file_name)
all.to_excel(save_file,
                 engine='openpyxl',
                 startrow=0,
                 startcol=0,
                 header=True,
                 na_rep='NaN',
                 float_format='%.2f',
                 sheet_name='Sheet1'
                 )


for num_topics in [3,4,5]:
    for iterations in [50,100,150,200,250,300]:
        for passes in [1,2,3,4,5]:

            lda_model= gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,
                                                       num_topics=num_topics,random_state=100,update_every=1,
                                                       chunksize=20000, passes=passes,iterations=iterations,
                                                       alpha='auto',per_word_topics=True)


            #compute perplexity
            lda_perplexity=lda_model.log_perplexity(corpus)
            print("S5, num_topics is %d, passes is %d and iterations is %d"%(num_topics,passes,iterations))
            print('\nPerplexity:',lda_perplexity)# a measure of how good the model is, lower the better.


            #compute coherence score
            coherence_model_lda=CoherenceModel(model=lda_model,texts=texts,dictionary=id2word,coherence='u_mass')
            coherence_lda=coherence_model_lda.get_coherence()
            print('\nCoherence Score:',coherence_lda)


            lda_model1= gensim.models.ldamodel.LdaModel(corpus=corpus1,id2word=id2word1,
                                                       num_topics=num_topics,random_state=100,update_every=1,
                                                       chunksize=20000, passes=passes,iterations=iterations,
                                                       alpha='auto',per_word_topics=True)



            #compute perplexity
            lda_perplexity1=lda_model1.log_perplexity(corpus1)
            print("S1,num_topics is %d, passes is %d and iterations is %d"%(num_topics,passes,iterations))
            print('\nPerplexity1:',lda_perplexity1)# a measure of how good the model is, lower the better.


            #compute coherence score
            coherence_model_lda1=CoherenceModel(model=lda_model1,texts=texts1,dictionary=id2word1,coherence='u_mass')
            coherence_lda1=coherence_model_lda1.get_coherence()
            print('\nCoherence Score1:',coherence_lda1)
