import pandas as pd
import numpy as np
import matplotlib as plt
import random

class EDA:
    """
    Data analysis object built on top of pandas Dataframe objects
    for exploratory anlysis and data preperation
    """
    def __init__(self, filepath='', delimiter=','):
        if len(filepath) == 0:
            self.data = self._test_data()
        else:
            self.data = pd.read_csv(filepath, delimiter)

        self.num_cols = [x for x in self.data.columns
                        if np.issubdtype(self.data[x].dtype, np.dtype(int).type)
                        or np.issubdtype(self.data[x].dtype, np.dtype(float).type)]

        self.exam = self.get_exam()
        self.shape = self.data.shape
        self.head = self.data.head()

    def get_exam(self, quant = .99):

        """
        Builds dataframe displaying important insights and information regarding
        a data set

        INPUTS
        Quantile: upper and lower bounds for counting outliers in a dataset

        OUTPUT
        Dataframe
        """
        self.exam = pd.DataFrame({
                            'Type': [str(self.data[x].dtype)
                                    for x in self.data.columns],
                            '% Missing': ((self.data.isnull().sum()/len(self.data))*100),

                            'Uniq Count': self.data.nunique(),

                            'Upper': [len(self.data[self.data[col] > self.data[col].quantile(quant)]) if col
                                     in self.num_cols
                                     else None
                                     for col in self.data.columns],

                            'Lower': [len(self.data[self.data[col] < self.data[col].quantile(1-quant)]) if col
                                     in self.num_cols
                                     else None
                                     for col in self.data.columns],
        })

        return self.exam

    def impute_missing(self, strategy='median'):
        """
        Method for quickly imputing missing values in a dataset

        INPUTS
        Strategy: Median, Mode, or numerical value to fill missing data with
        """
        for col in self.num_cols:

            if strategy == 'median':
                self.data[col].fillna((self.data[col].mean()), inplace=True)

            if strategy == 'mode':
                self.data[col].fillna((self.data[col].mode()), inplace=True)

            else:
                self.data[col].fillna((strategy), inplace=True)

        self._update_attrs()

    def create_dummies(self, column):
        """
        Creates dummy variables for selected column of dataset, drops original column and concats
        to original dataset

        INPUT
        Column: Select column of dataset to one hot encode
        """
        dummy_col = pd.get_dummies(self.data[column])
        dummy_col = dummy_col.add_prefix(column+'_')

        self.data = pd.concat([self.data, dummy_col], axis=1)
        self.data = self.data.drop(column, axis=1)

        self._update_attrs()

    def thread(self, column):
        """
        Method for displaying descriptive statistics, data visualizations, etc. for selected
        column of dataset

        INPUT
        Column: selected column to perform analysis

        OUTPUT:
        -Histrogram
        -Correlations Series
        -Pandas decribe() Series
        """
        self.data[column].hist();
        print (self.data[column].describe())
        return self.data.corr()[column]

    def _update_attrs(self):

        self.num_cols = [x for x in self.data.columns
                        if np.issubdtype(self.data[x].dtype, np.dtype(int).type)
                        or np.issubdtype(self.data[x].dtype, np.dtype(float).type)]

        self.exam = self.get_exam()
        self.shape = self.data.shape
        self.head = self.data.head()

    def _test_data(self):
        """
        Creates sample dataframe when filepath is not provided
        """
        test_df = pd.DataFrame(np.random.randn(50, 3), columns=list('ABC'))
        test_df['D'] = np.random.choice([1, 2, 3], test_df.shape[0])
        test_df['E'] = np.where((test_df['B']< 0), np.nan, np.random.choice([1, 2, 3], test_df.shape[0]))
        self.data = test_df
        return self.data;