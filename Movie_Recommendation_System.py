#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import numpy as np


# In[27]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[28]:


movies.head()


# In[29]:


credits.head()


# In[30]:


movies = movies.merge(credits,on='title')


# In[31]:


movies.shape


# In[33]:


credits.shape


# In[34]:


movies.head()


# In[35]:


movies.shape


# In[36]:


movies.info()


# In[12]:


# genres
# id
# keywords
# overview
# title
# cast
# crew


# In[37]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[38]:


movies.head()


# In[39]:


movies.isnull().sum()


# In[40]:


movies.dropna(inplace=True)


# In[42]:


movies.duplicated().sum()


# In[43]:


movies.iloc[0].genres


# In[19]:


#'[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]
#['Action','Adventure','Fantancy','Science Fiction']


# In[52]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[54]:


import ast
ast.literal_eval


# In[56]:


movies['genres'] = movies['genres'].apply(convert)


# In[57]:


movies.head()


# In[58]:


movies.iloc[0].keywords


# In[60]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[61]:


movies.head()


# In[62]:


movies.cast[0]


# In[63]:


def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[64]:


movies['cast'] = movies['cast'].apply(convert3)


# In[65]:


movies.head()


# In[66]:


movies.crew[0]


# In[67]:


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[68]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[70]:


movies.head()


# In[71]:


movies.overview[0]


# In[72]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[73]:


movies.head()


# In[74]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])


# In[75]:


movies.head()


# In[76]:


movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])


# In[77]:


movies.head()


# In[78]:


movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])


# In[79]:


movies.head()


# In[80]:


movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[81]:


movies.head()


# In[82]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[83]:


movies.head()


# In[87]:


new_df = movies[['movie_id', 'title', 'tags']]


# In[88]:


new_df


# In[97]:


new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))


# In[98]:


new_df.head()


# In[140]:


import nltk


# In[141]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[142]:


def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)  


# In[146]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[147]:


new_df['tags'][0]


# In[148]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[149]:


new_df.head()


# In[150]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')


# In[151]:


vector = cv.fit_transform(new_df['tags']).toarray()


# In[152]:


vector


# In[153]:


vector[0]


# In[172]:


cv.get_feature_names_out()


# In[173]:


from sklearn.metrics.pairwise import cosine_similarity


# In[174]:


similarity = cosine_similarity(vectors)


# In[176]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[165]:


similarity


# In[184]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[0]]['title'])


# In[187]:


recommend('Batman Begins')


# In[ ]:




