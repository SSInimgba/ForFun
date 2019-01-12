
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[2]:


pd.set_option('display.max_colwidth', -1)
df = pd.read_csv('wizkid_lyrics.csv', usecols = [0,1,2,3,4,5,6,7])
df = df.dropna(how = 'all')
df.head(1)


# In[3]:


df.to_csv('wizkidslyricsraw.csv')
df.head(1)


# In[4]:


df.describe()


# In[5]:


df.dtypes


# In[6]:


df[['Lyrics']]


# In[7]:


#Cleaning up the lyrics text a bit
#regex is a pain in the butt but pythex.org helps alot!
df['Lyrics'] = df.Lyrics.replace('\n', ' ', regex= True)#removes newline
df['Lyrics'] = df.Lyrics.replace('\(|\)', ' ', regex= True)#removes brackets
df['Lyrics'] = df.Lyrics.replace('\[[^]]*.', ' ', regex= True)#removes square brackets and text in them
df['Lyrics'] = df.Lyrics.replace('\{(.*?)\}', ' ', regex= True)#removes curly brackets and text in them
df['Lyrics'] = df.Lyrics.replace('\?', ' ', regex= True)#removes question marks
df['Lyrics']=  df.Lyrics.replace('(\...)|(\..)', ' ', regex= True)#removes '...' marks
df[['Lyrics']].head(1)


# In[8]:


df.head(1)


# In[9]:


#stopwords to remove noise from lyrics
import string
import nltk
from nltk.corpus import stopwords
from nltk import FreqDist

punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation +  ['', '@ ','&', 'x2', 
                                                    '2x','go','dey','say','na','see', 'oya',
                                                   'lo','le', 'e', "i'm",'like','ni','let'] 
text = df['Lyrics']

#tokenize the tweet text
tokens = []
for txt in text.values:
    tokens.extend([t.lower().strip(":,.") for t in txt.split()])
    
filtered_tokens = [w for w in tokens if not w in stop]
freq_dist = nltk.FreqDist(filtered_tokens)
freq_dist.most_common(10000)


# In[10]:


#preparing x and y axes to be ploted 
XY = freq_dist.items()
#pair x and y and sort the axes
XY = sorted (XY, key=lambda pair: pair[1], reverse= True)
XY.sort(key = lambda pair: pair[1], reverse = True)
limit = 30
#unpair the axes
X = [x for (x,y) in XY[:limit]]
Y = [y for (x,y) in XY[:limit]]
#Transform X into a list of numbers from a list of word tokens
range(4)
nX = range(len(X))
nX



# In[11]:


#First plot of wizkids most used words in his lyrics
plt.figure(figsize =(20,15))
plt.plot(nX,Y, linewidth = 5)
plt.xticks(nX,X, rotation = 'vertical', fontsize = 25 )
plt.yticks(fontsize = 29 )
plt.xlabel('Tokens', fontsize = 20);
plt.ylabel('Occurances', fontsize = 25);
plt.title("Most used words in Wizkid's lyrics", fontsize = 35)
plt.tight_layout();
plt.show();


# In[12]:


df.dtypes


# In[13]:


#Group lyrics by Albums
dfgrouped = df.groupby('Album')[['Lyrics']].sum()
dfgrouped['Lyrics'] = dfgrouped['Lyrics'].str.lower()
dfgrouped.head(1)


# In[14]:


#count the occurence of love in each album
dfgrouped['Love_occurance'] = dfgrouped.Lyrics.str.count('love')
dfgrouped['Love_occurance']


# In[15]:


dfgrouped['Love_occurance'].plot(kind = 'bar', figsize =(18,7), linewidth = 2, fontsize = 12);
plt.xlabel('Album', fontsize = 15  );
plt.xticks(rotation = 0, fontsize = 15);
plt.yticks(fontsize = 15);
plt.title("How 'love' was mentioned across Wizkid's Albums", fontsize = 20);
plt.ylabel('Number of times Love was mentioned', fontsize = 20);
plt.show();


# In[16]:


#count the occurence of baby in each album
dfgrouped['baby_occurance'] = dfgrouped.Lyrics.str.count('baby')
dfgrouped['baby_occurance']


# In[17]:


dfgrouped['baby_occurance'].plot(kind = 'bar', figsize =(18,7), linewidth = 2, fontsize = 12);
plt.xlabel('Album', fontsize = 15  );
plt.xticks(rotation = 0, fontsize = 15 );
plt.yticks(rotation = 0, fontsize = 15 );
plt.title("How 'baby' was mentioned across Wizkid's Albums", fontsize = 20);
plt.ylabel('Count', fontsize = 20);
plt.show();


# In[18]:


df.groupby('Album')[['Tracks']].sum()


# In[19]:


#count the occurence of starboy in each album
dfgrouped['starboy_occurance'] = dfgrouped.Lyrics.str.count('starboy')
dfgrouped['starboy_occurance']


# In[20]:


dfgrouped['starboy_occurance'].plot(kind = 'bar', figsize =(18,7), linewidth = 2, fontsize = 12);
plt.xlabel('Album', fontsize = 15  );
plt.xticks(rotation = 0);
plt.title("How 'Starboy' was mentioned across Wizkid's Albums", fontsize = 20);
plt.ylabel('Count', fontsize = 12);
plt.show();


# In[21]:


#count the occurence of wizzy in each album
dfgrouped['wizzy_occurance'] = dfgrouped.Lyrics.str.count('wizzy')
dfgrouped['wizzy_occurance']


# In[22]:


dfgrouped['wizzy_occurance'].plot(kind = 'bar', figsize =(18,7), linewidth = 2, fontsize = 12);
plt.xlabel('Album', fontsize = 15  );
plt.xticks(rotation = 0);
plt.title("How 'Wizzy' was mentioned across Wizkid's Albums", fontsize = 20);
plt.ylabel('Count', fontsize = 12);
plt.show();


# In[23]:


#Order the way the albums are displayed
album_order = ['Superstar','Ayo', 'Sounds From The Other Side']
dfgrouped = dfgrouped.loc[album_order]


# In[24]:


fig, ax = plt.subplots(sharex = True, figsize =(25,10),  linewidth = 2);
ax.plot(dfgrouped.index, dfgrouped[['wizzy_occurance', 'starboy_occurance']] );
ax.legend(["Wizzy", "Starboy"], loc ='upper right',fontsize = 25)
ax.xaxis.labelpad = 20; #add padding to x axis
ax.yaxis.labelpad = 20;
plt.xlabel('Album', fontsize = 30);
plt.xticks(rotation = 0, fontsize = 25);
plt.yticks(fontsize = 25 )
plt.title("Progression of the mentions  of 'Wizzy' and 'Starboy' across albums", fontsize = 30, pad = 20);
plt.ylabel('Count', fontsize = 30);
sns.set_style("darkgrid", { "xtick.color": ".5"})
sns.set_style({ "ytick.color": ".5"})
sns.set_context("talk");
plt.show();


# In[25]:


#show index of dfgrouped dataframe
dfgrouped.index


# In[26]:


#uniqueness score for each song(S) = unique words in song(s) divided by total words in song(t)
#uniqueness score for album = average of S
df_uniqueness_score = dfgrouped.Lyrics.str.count('[^\s]') #count every word in the album
df_uniqueness_score


# In[27]:


df_tracks = df.groupby(['Album','Tracks'])[['Lyrics']].sum()
df_tracks


# In[28]:


#uniqueness score for each song(S) = unique words in song(s) divided by total words in song(t)
#uniqueness score for album = average of S
totalscore = df_tracks.Lyrics.str.count('[^\s]') #count every word in a track
totalscore


# In[29]:


import collections
from collections import Counter
results = Counter()
test = df_tracks.Lyrics.str.lower().str.split().apply(results.update)
unique_counts = sum((results).values())
test


# In[30]:


#find out what is stored in a variable
type(totalscore)


# In[31]:


sum(Counter(['way','way', 'you', 'whine', 'your', 'body']).values()) #test to see how counter works


# In[32]:


#count the unique values in lyrics
sum((results).values())


# In[33]:


df_tracks.dtypes


# In[34]:


#uniqueness score for each song(S) = unique words in song(s) divided by total words in song(t)
#uniqueness score for album = average of S
totalscore = df_tracks.Lyrics.str.count('[^\s]') #count every word in a track
totalscore


# In[35]:


df_test = df_tracks 


# In[45]:


#wordcount of lyrics and unique word count of lyrics
df_test['Lyrics_tokenized'] = df_test['Lyrics'].str.split()
df_test['LyricsCounter'] = df_test['Lyrics_tokenized'].apply(Counter) #create counter dictionary
df_test['LyricsWords'] = df_test['Lyrics_tokenized'].apply(len)
df_test['LyricsUniqueWords'] = df_test['LyricsCounter'].apply(len)
df_test.head(1)


# In[37]:



df_test.index


# In[38]:


#uniqueness score for each song(S) = unique words in song(s) divided by total words in song(t)
#uniqueness score for album = average of S
df_test['unique_score'] = df_test['LyricsUniqueWords'] / df_test['LyricsWords']
df_test.dtypes


# In[53]:


#Unique score of each track
df_tracks = df_test[['unique_score']].groupby(by=['Tracks']).mean()
df_tracks.head(5)


# In[40]:


#Unique score of each album
df_uniquescore = df_test[['unique_score']].groupby(by=['Album']).mean()
df_uniquescore


# In[41]:


#Order the way the albums are displayed
album_order = ['Superstar','Ayo', 'Sounds From The Other Side']
df_uniquescore = df_uniquescore.loc[album_order]
df_uniquescore


# In[42]:


df_uniquescore.plot(kind = 'bar', figsize =(18,7), linewidth = 2, fontsize = 12);
plt.xticks(rotation = 0);
plt.title("Vocabulary of Albums", fontsize = 20,  pad = 20);
plt.xlabel('Album', fontsize = 15  );
plt.ylabel('Uniqueness Score', fontsize = 12);
plt.show();


# Tracks in Superstar = 17
# Tracks in Ayo = 19
# Tracks in SFTOS = 12

# In[56]:


#vocabulary of tracks
df_tracks.sort_values('unique_score', ascending = True).plot(kind = 'barh', figsize =(18,20), linewidth = 2, fontsize = 12);

