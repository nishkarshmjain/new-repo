# Simple Approaches to Recommender Systems
## Making Recommendations Based on Correlation


```python
import numpy as np
import pandas as pd
```

These datasets are hosted on: https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data

They were originally published by: Blanca Vargas-Govea, Juan Gabriel GonzÃ¡lez-Serna, Rafael Ponce-MedellÃ­n. Effects of relevant contextual features in the performance of a restaurant recommender system. In RecSysâ€™11: Workshop on Context Aware Recommender Systems (CARS-2011), Chicago, IL, USA, October 23, 2011.


```python
frame =  pd.read_csv('rating_final.csv')
cuisine = pd.read_csv('chefmozcuisine.csv')
geodata = pd.read_csv('geoplaces2.csv', encoding = 'mbcs')
```


```python
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
      <th>userID</th>
      <th>placeID</th>
      <th>rating</th>
      <th>food_rating</th>
      <th>service_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>U1077</td>
      <td>135085</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>U1077</td>
      <td>135038</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>U1077</td>
      <td>132825</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>U1077</td>
      <td>135060</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>U1068</td>
      <td>135104</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
geodata.head()
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
      <th>placeID</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>the_geom_meter</th>
      <th>name</th>
      <th>address</th>
      <th>city</th>
      <th>state</th>
      <th>country</th>
      <th>fax</th>
      <th>...</th>
      <th>alcohol</th>
      <th>smoking_area</th>
      <th>dress_code</th>
      <th>accessibility</th>
      <th>price</th>
      <th>url</th>
      <th>Rambience</th>
      <th>franchise</th>
      <th>area</th>
      <th>other_services</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>134999</td>
      <td>18.915421</td>
      <td>-99.184871</td>
      <td>0101000020957F000088568DE356715AC138C0A525FC46...</td>
      <td>Kiku Cuernavaca</td>
      <td>Revolucion</td>
      <td>Cuernavaca</td>
      <td>Morelos</td>
      <td>Mexico</td>
      <td>?</td>
      <td>...</td>
      <td>No_Alcohol_Served</td>
      <td>none</td>
      <td>informal</td>
      <td>no_accessibility</td>
      <td>medium</td>
      <td>kikucuernavaca.com.mx</td>
      <td>familiar</td>
      <td>f</td>
      <td>closed</td>
      <td>none</td>
    </tr>
    <tr>
      <th>1</th>
      <td>132825</td>
      <td>22.147392</td>
      <td>-100.983092</td>
      <td>0101000020957F00001AD016568C4858C1243261274BA5...</td>
      <td>puesto de tacos</td>
      <td>esquina santos degollado y leon guzman</td>
      <td>s.l.p.</td>
      <td>s.l.p.</td>
      <td>mexico</td>
      <td>?</td>
      <td>...</td>
      <td>No_Alcohol_Served</td>
      <td>none</td>
      <td>informal</td>
      <td>completely</td>
      <td>low</td>
      <td>?</td>
      <td>familiar</td>
      <td>f</td>
      <td>open</td>
      <td>none</td>
    </tr>
    <tr>
      <th>2</th>
      <td>135106</td>
      <td>22.149709</td>
      <td>-100.976093</td>
      <td>0101000020957F0000649D6F21634858C119AE9BF528A3...</td>
      <td>El Rincón de San Francisco</td>
      <td>Universidad 169</td>
      <td>San Luis Potosi</td>
      <td>San Luis Potosi</td>
      <td>Mexico</td>
      <td>?</td>
      <td>...</td>
      <td>Wine-Beer</td>
      <td>only at bar</td>
      <td>informal</td>
      <td>partially</td>
      <td>medium</td>
      <td>?</td>
      <td>familiar</td>
      <td>f</td>
      <td>open</td>
      <td>none</td>
    </tr>
    <tr>
      <th>3</th>
      <td>132667</td>
      <td>23.752697</td>
      <td>-99.163359</td>
      <td>0101000020957F00005D67BCDDED8157C1222A2DC8D84D...</td>
      <td>little pizza Emilio Portes Gil</td>
      <td>calle emilio portes gil</td>
      <td>victoria</td>
      <td>tamaulipas</td>
      <td>?</td>
      <td>?</td>
      <td>...</td>
      <td>No_Alcohol_Served</td>
      <td>none</td>
      <td>informal</td>
      <td>completely</td>
      <td>low</td>
      <td>?</td>
      <td>familiar</td>
      <td>t</td>
      <td>closed</td>
      <td>none</td>
    </tr>
    <tr>
      <th>4</th>
      <td>132613</td>
      <td>23.752903</td>
      <td>-99.165076</td>
      <td>0101000020957F00008EBA2D06DC8157C194E03B7B504E...</td>
      <td>carnitas_mata</td>
      <td>lic. Emilio portes gil</td>
      <td>victoria</td>
      <td>Tamaulipas</td>
      <td>Mexico</td>
      <td>?</td>
      <td>...</td>
      <td>No_Alcohol_Served</td>
      <td>permitted</td>
      <td>informal</td>
      <td>completely</td>
      <td>medium</td>
      <td>?</td>
      <td>familiar</td>
      <td>t</td>
      <td>closed</td>
      <td>none</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
places =  geodata[['placeID', 'name']]
places.head()
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
      <th>placeID</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>134999</td>
      <td>Kiku Cuernavaca</td>
    </tr>
    <tr>
      <th>1</th>
      <td>132825</td>
      <td>puesto de tacos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>135106</td>
      <td>El Rincón de San Francisco</td>
    </tr>
    <tr>
      <th>3</th>
      <td>132667</td>
      <td>little pizza Emilio Portes Gil</td>
    </tr>
    <tr>
      <th>4</th>
      <td>132613</td>
      <td>carnitas_mata</td>
    </tr>
  </tbody>
</table>
</div>




```python
cuisine.head()
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
      <th>placeID</th>
      <th>Rcuisine</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>135110</td>
      <td>Spanish</td>
    </tr>
    <tr>
      <th>1</th>
      <td>135109</td>
      <td>Italian</td>
    </tr>
    <tr>
      <th>2</th>
      <td>135107</td>
      <td>Latin_American</td>
    </tr>
    <tr>
      <th>3</th>
      <td>135106</td>
      <td>Mexican</td>
    </tr>
    <tr>
      <th>4</th>
      <td>135105</td>
      <td>Fast_Food</td>
    </tr>
  </tbody>
</table>
</div>



## Grouping and Ranking Data


```python
rating = pd.DataFrame(frame.groupby('placeID')['rating'].mean())
rating.head()
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
      <th>rating</th>
    </tr>
    <tr>
      <th>placeID</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>132560</th>
      <td>0.50</td>
    </tr>
    <tr>
      <th>132561</th>
      <td>0.75</td>
    </tr>
    <tr>
      <th>132564</th>
      <td>1.25</td>
    </tr>
    <tr>
      <th>132572</th>
      <td>1.00</td>
    </tr>
    <tr>
      <th>132583</th>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
rating['rating_count'] = pd.DataFrame(frame.groupby('placeID')['rating'].count())
rating.head()
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
      <th>rating</th>
      <th>rating_count</th>
    </tr>
    <tr>
      <th>placeID</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>132560</th>
      <td>0.50</td>
      <td>4</td>
    </tr>
    <tr>
      <th>132561</th>
      <td>0.75</td>
      <td>4</td>
    </tr>
    <tr>
      <th>132564</th>
      <td>1.25</td>
      <td>4</td>
    </tr>
    <tr>
      <th>132572</th>
      <td>1.00</td>
      <td>15</td>
    </tr>
    <tr>
      <th>132583</th>
      <td>1.00</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
rating.describe()
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
      <th>rating</th>
      <th>rating_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>130.000000</td>
      <td>130.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.179622</td>
      <td>8.930769</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.349354</td>
      <td>6.124279</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.250000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.181818</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.400000</td>
      <td>11.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.000000</td>
      <td>36.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
rating.sort_values('rating_count', ascending=False).head()
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
      <th>rating</th>
      <th>rating_count</th>
    </tr>
    <tr>
      <th>placeID</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>135085</th>
      <td>1.333333</td>
      <td>36</td>
    </tr>
    <tr>
      <th>132825</th>
      <td>1.281250</td>
      <td>32</td>
    </tr>
    <tr>
      <th>135032</th>
      <td>1.178571</td>
      <td>28</td>
    </tr>
    <tr>
      <th>135052</th>
      <td>1.280000</td>
      <td>25</td>
    </tr>
    <tr>
      <th>132834</th>
      <td>1.000000</td>
      <td>25</td>
    </tr>
  </tbody>
</table>
</div>




```python
places[places['placeID']==135085]
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
      <th>placeID</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>121</th>
      <td>135085</td>
      <td>Tortas Locas Hipocampo</td>
    </tr>
  </tbody>
</table>
</div>




```python
cuisine[cuisine['placeID']==135085]
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
      <th>placeID</th>
      <th>Rcuisine</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>44</th>
      <td>135085</td>
      <td>Fast_Food</td>
    </tr>
  </tbody>
</table>
</div>



## Preparing Data For Analysis


```python
places_crosstab = pd.pivot_table(data=frame, values='rating', index='userID', columns='placeID')
places_crosstab.head()
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
      <th>placeID</th>
      <th>132560</th>
      <th>132561</th>
      <th>132564</th>
      <th>132572</th>
      <th>132583</th>
      <th>132584</th>
      <th>132594</th>
      <th>132608</th>
      <th>132609</th>
      <th>132613</th>
      <th>...</th>
      <th>135080</th>
      <th>135081</th>
      <th>135082</th>
      <th>135085</th>
      <th>135086</th>
      <th>135088</th>
      <th>135104</th>
      <th>135106</th>
      <th>135108</th>
      <th>135109</th>
    </tr>
    <tr>
      <th>userID</th>
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
      <th>U1001</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>U1002</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>U1003</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>U1004</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>U1005</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 130 columns</p>
</div>




```python
Tortas_ratings = places_crosstab[135085]
Tortas_ratings[Tortas_ratings>=0]
```




    userID
    U1001    0.0
    U1002    1.0
    U1007    1.0
    U1013    1.0
    U1016    2.0
    U1027    1.0
    U1029    1.0
    U1032    1.0
    U1033    2.0
    U1036    2.0
    U1045    2.0
    U1046    1.0
    U1049    0.0
    U1056    2.0
    U1059    2.0
    U1062    0.0
    U1077    2.0
    U1081    1.0
    U1084    2.0
    U1086    2.0
    U1089    1.0
    U1090    2.0
    U1092    0.0
    U1098    1.0
    U1104    2.0
    U1106    2.0
    U1108    1.0
    U1109    2.0
    U1113    1.0
    U1116    2.0
    U1120    0.0
    U1122    2.0
    U1132    2.0
    U1134    2.0
    U1135    0.0
    U1137    2.0
    Name: 135085, dtype: float64




```python

```

## Evaluating Similarity Based on Correlation


```python
similar_to_Tortas = places_crosstab.corrwith(Tortas_ratings)

corr_Tortas = pd.DataFrame(similar_to_Tortas, columns=['PearsonR'])
corr_Tortas.dropna(inplace=True)
corr_Tortas.head()
```

    C:\Users\piers\Anaconda3\lib\site-packages\numpy\lib\function_base.py:2995: RuntimeWarning: Degrees of freedom <= 0 for slice
      c = cov(x, y, rowvar)
    C:\Users\piers\Anaconda3\lib\site-packages\numpy\lib\function_base.py:2929: RuntimeWarning: divide by zero encountered in double_scalars
      c *= 1. / np.float64(fact)
    




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
      <th>PearsonR</th>
    </tr>
    <tr>
      <th>placeID</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>132572</th>
      <td>-0.428571</td>
    </tr>
    <tr>
      <th>132723</th>
      <td>0.301511</td>
    </tr>
    <tr>
      <th>132754</th>
      <td>0.930261</td>
    </tr>
    <tr>
      <th>132825</th>
      <td>0.700745</td>
    </tr>
    <tr>
      <th>132834</th>
      <td>0.814823</td>
    </tr>
  </tbody>
</table>
</div>




```python
Tortas_corr_summary = corr_Tortas.join(rating['rating_count'])
```


```python
Tortas_corr_summary[Tortas_corr_summary['rating_count']>=10].sort_values('PearsonR', ascending=False).head(10)
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
      <th>PearsonR</th>
      <th>rating_count</th>
    </tr>
    <tr>
      <th>placeID</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>135076</th>
      <td>1.000000</td>
      <td>13</td>
    </tr>
    <tr>
      <th>135085</th>
      <td>1.000000</td>
      <td>36</td>
    </tr>
    <tr>
      <th>135066</th>
      <td>1.000000</td>
      <td>12</td>
    </tr>
    <tr>
      <th>132754</th>
      <td>0.930261</td>
      <td>13</td>
    </tr>
    <tr>
      <th>135045</th>
      <td>0.912871</td>
      <td>13</td>
    </tr>
    <tr>
      <th>135062</th>
      <td>0.898933</td>
      <td>21</td>
    </tr>
    <tr>
      <th>135028</th>
      <td>0.892218</td>
      <td>15</td>
    </tr>
    <tr>
      <th>135042</th>
      <td>0.881409</td>
      <td>20</td>
    </tr>
    <tr>
      <th>135046</th>
      <td>0.867722</td>
      <td>11</td>
    </tr>
    <tr>
      <th>132872</th>
      <td>0.840168</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




```python
places_corr_Tortas = pd.DataFrame([135085, 132754, 135045, 135062, 135028, 135042, 135046], index = np.arange(7), columns=['placeID'])
summary = pd.merge(places_corr_Tortas, cuisine,on='placeID')
summary
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
      <th>placeID</th>
      <th>Rcuisine</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>135085</td>
      <td>Fast_Food</td>
    </tr>
    <tr>
      <th>1</th>
      <td>132754</td>
      <td>Mexican</td>
    </tr>
    <tr>
      <th>2</th>
      <td>135028</td>
      <td>Mexican</td>
    </tr>
    <tr>
      <th>3</th>
      <td>135042</td>
      <td>Chinese</td>
    </tr>
    <tr>
      <th>4</th>
      <td>135046</td>
      <td>Fast_Food</td>
    </tr>
  </tbody>
</table>
</div>




```python
places[places['placeID']==135046]
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
      <th>placeID</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42</th>
      <td>135046</td>
      <td>Restaurante El Reyecito</td>
    </tr>
  </tbody>
</table>
</div>




```python
cuisine['Rcuisine'].describe()
```




    count         916
    unique         59
    top       Mexican
    freq          239
    Name: Rcuisine, dtype: object




```python

```


```python

```
