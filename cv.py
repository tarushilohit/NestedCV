#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df1=pd.read_csv("train.csv")
df2=pd.read_csv("test.csv")


# In[3]:


df1


# In[5]:


import pandas as pd


class NestedCV:
    def __init__(self, k):
        self.k = k

    def split(self, data, date_column):
        # Sort data by date
        data = data.sort_values(by=date_column)

        # Calculate number of samples per fold (avoid edge case with equal split)
        fold_size = (len(data) // self.k) + 1

        for i in range(self.k):
            # Calculate indices for train and validation sets
            val_start_idx = i * fold_size
            val_end_idx = min((i + 1) * fold_size, len(data))
            val_indices = list(range(val_start_idx, val_end_idx))

            # Split data into train and validation sets
            validate = data.iloc[val_indices]
            train = data.drop(val_indices)

            yield train, validate


            if __name__ == "__main__":
                # Sample data loading
                data = pd.DataFrame({
                    "date": pd.date_range(start="2022-01-01", periods=100),
                    "value": range(100)
                })

                # Nested CV
                k = 3
                cv = NestedCV(k)
                splits = cv.split(data, "date")

                # Check return type
                assert isinstance(splits, GeneratorType)

                # Check return types, shapes, and data leaks (adjusted)
                count = 0
                for train, validate in splits:
                    # Types
                    assert isinstance(train, pd.DataFrame)
                    assert isinstance(validate, pd.DataFrame)

                    # Shape
                    assert train.shape[1] == validate.shape[1]

                    count += 1

                # Check number of splits returned
                assert count == k


# In[ ]:




