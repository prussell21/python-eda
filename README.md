# python-eda

Work in Progress: 
Python package built on top of the Pandas library for increasing efficiency and effectiveness of exploratory data analysis, data preparation for machine learning, performing inference, and gaining insight.

## Requirements

```
import pandas
import numpy
import matplotlib
from sklearn.model_selection import train_test_split
```

## Usage

EDA objects accept csv filepath or existing DataFrame object as input, or creates a default test set if no input is provided

```
#filepath
eda = EDA('filename.csv')

#dataframe
eda = EDA(df)

#generated dataframe
eda = EDA()
```
Access data from original dataframe with .data attribute

```
eda.data.head()
```

### Methods

```
#Examine dataset
eda.examine()

#Impute missing values. Strategies = median, mode, or numerical value
eda.impute_missing(strategy = 'median')

#create dummy variable columns
eda.create_dummies(column=input_column)

#Split target and feature into train/test sets for training machine learning models
eda.split(target=target_column)

#View statistics and distribution of selected column
eda.dashboard(column)
```
