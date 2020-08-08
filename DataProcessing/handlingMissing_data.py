import pandas as pd
import numpy as np

dataset = pd.read_csv('Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
#print(x)

new=dataset.iloc[:,:-1]
print(new['Salary'])

from sklearn.impute import SimpleImputer as SI
imputer = SI(missing_values=np.nan, strategy='mean')
x[:,1:3]=imputer.fit_transform(x[:,1:3])
new = new.fillna(new.mean())
#imputer.fit(x[:,1:3])
#x[:,1:3]=imputer.transform(x[:,1:3])
print(x)
#print(new)