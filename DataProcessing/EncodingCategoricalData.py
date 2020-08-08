import pandas as pd
import numpy as np

dataset = pd.read_csv('Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.impute import SimpleImputer as SI
imputer = SI(missing_values=np.nan, strategy='mean')
x[:,1:3]=imputer.fit_transform(x[:,1:3])

from sklearn.compose import  ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers= [('encoder',OneHotEncoder(),[0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x)
from sklearn.preprocessing import LabelEncoder

#lex= LabelEncoder()
#x[:,0]=lex.fit_transform(x[:,0])
#print(x)

le= LabelEncoder()
y=le.fit_transform(y)
print(y)