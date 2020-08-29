---
title: "Collaborative Filtering-Based Recommender"
date: 2020-03-20
tags: [Machine Learning, Data Science, Recommendation Engine]
header:
  image: "/images/rc1.jpg"
excerpt: "Machine Learning, Data Science, Recommendation Engine "
mathjax: "true"
---

## Machine Learning Based Recommendation Systems
## SVD Matrix Factorization


```python
import numpy as np
import pandas as pd

import sklearn
from sklearn.decomposition import TruncatedSVD
```

The MovieLens dataset was collected by the GroupLens Research Project at the University of Minnesota. You can download the dataset for this demostration at the following URL: https://grouplens.org/datasets/movielens/100k/

### Preparing the data


```python
columns = ['user_id', 'item_id', 'rating', 'timestamp']
frame = pd.read_csv('ml-100k/u.data', sep='\t', names=columns)
frame.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>item_id</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>196</td>
      <td>242</td>
      <td>3</td>
      <td>881250949</td>
    </tr>
    <tr>
      <th>1</th>
      <td>186</td>
      <td>302</td>
      <td>3</td>
      <td>891717742</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>377</td>
      <td>1</td>
      <td>878887116</td>
    </tr>
    <tr>
      <th>3</th>
      <td>244</td>
      <td>51</td>
      <td>2</td>
      <td>880606923</td>
    </tr>
    <tr>
      <th>4</th>
      <td>166</td>
      <td>346</td>
      <td>1</td>
      <td>886397596</td>
    </tr>
  </tbody>
</table>
</div>




```python
columns = ['item_id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
          'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movies = pd.read_csv('ml-100k/u.item', sep='|', names=columns, encoding='latin-1')
movie_names = movies[['item_id', 'movie title']]
movie_names.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item_id</th>
      <th>movie title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>GoldenEye (1995)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Four Rooms (1995)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Get Shorty (1995)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Copycat (1995)</td>
    </tr>
  </tbody>
</table>
</div>




```python
combined_movies_data = pd.merge(frame, movie_names, on='item_id')
combined_movies_data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>item_id</th>
      <th>rating</th>
      <th>timestamp</th>
      <th>movie title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>196</td>
      <td>242</td>
      <td>3</td>
      <td>881250949</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>63</td>
      <td>242</td>
      <td>3</td>
      <td>875747190</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>226</td>
      <td>242</td>
      <td>5</td>
      <td>883888671</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>154</td>
      <td>242</td>
      <td>3</td>
      <td>879138235</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>306</td>
      <td>242</td>
      <td>5</td>
      <td>876503793</td>
      <td>Kolya (1996)</td>
    </tr>
  </tbody>
</table>
</div>




```python
combined_movies_data.groupby('item_id')['rating'].count().sort_values(ascending=False).head()
```




    item_id
    50     583
    258    509
    100    508
    181    507
    294    485
    Name: rating, dtype: int64




```python
filter = combined_movies_data['item_id']==50
combined_movies_data[filter]['movie title'].unique()
```




    array(['Star Wars (1977)'], dtype=object)



### Building a Utility Matrix


```python
rating_crosstab = combined_movies_data.pivot_table(values='rating', index='user_id', columns='movie title', fill_value=0)
rating_crosstab.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>movie title</th>
      <th>'Til There Was You (1997)</th>
      <th>1-900 (1994)</th>
      <th>101 Dalmatians (1996)</th>
      <th>12 Angry Men (1957)</th>
      <th>187 (1997)</th>
      <th>2 Days in the Valley (1996)</th>
      <th>20,000 Leagues Under the Sea (1954)</th>
      <th>2001: A Space Odyssey (1968)</th>
      <th>3 Ninjas: High Noon At Mega Mountain (1998)</th>
      <th>39 Steps, The (1935)</th>
      <th>...</th>
      <th>Yankee Zulu (1994)</th>
      <th>Year of the Horse (1997)</th>
      <th>You So Crazy (1994)</th>
      <th>Young Frankenstein (1974)</th>
      <th>Young Guns (1988)</th>
      <th>Young Guns II (1990)</th>
      <th>Young Poisoner's Handbook, The (1995)</th>
      <th>Zeus and Roxanne (1997)</th>
      <th>unknown</th>
      <th>Á köldum klaka (Cold Fever) (1994)</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1664 columns</p>
</div>



### Transposing the Matrix


```python
rating_crosstab.shape
```




    (943, 1664)




```python
X = rating_crosstab.T
X.shape
```




    (1664, 943)



### Decomposing the Matrix


```python
SVD = TruncatedSVD(n_components=12, random_state=17)

resultant_matrix = SVD.fit_transform(X)

resultant_matrix.shape
```




    (1664, 12)




```python

```

### Generating a Correlation Matrix


```python
corr_mat = np.corrcoef(resultant_matrix)
corr_mat.shape
```




    (1664, 1664)



### Isolating Star Wars From the Correlation Matrix


```python
movie_names = rating_crosstab.columns
movies_list = list(movie_names)

star_wars = movies_list.index('Star Wars (1977)')
star_wars
```




    1398




```python
corr_star_wars = corr_mat[1398]
corr_star_wars.shape
```




    (1664,)



### Recommending a Highly Correlated Movie


```python
list(movie_names[(corr_star_wars<1.0) & (corr_star_wars > 0.9)])

```




    ['Die Hard (1988)',
     'Empire Strikes Back, The (1980)',
     'Fugitive, The (1993)',
     'Raiders of the Lost Ark (1981)',
     'Return of the Jedi (1983)',
     'Terminator 2: Judgment Day (1991)',
     'Terminator, The (1984)',
     'Toy Story (1995)']




```python
list(movie_names[(corr_star_wars<1.0) & (corr_star_wars > 0.95)])
```




    ['Return of the Jedi (1983)']


