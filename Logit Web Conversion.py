
# coding: utf-8

# ## Predicting Conversions from Web Quotes

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import pyodbc
from sklearn import linear_model
import random


# Import the data from SQL Server

# In[2]:


query = """
select DISTINCT
	pp.Id as QuotePetLead
, isnull(iif(pp.AgeId = 33, 0.5, pp.AgeId - 0.5), 3.5) as age
, isnull(bf.weight, iif(breed.Name like '%Mixed breed Small%', 15,
						iif(breed.Name like 'Mixed breed Medium%', 38,
							iif(breed.Name like '%Mixed breed Large%', 73, 
								iif(breed.Name like '%Mixed breed Giant%', 110, 
									iif(breed.AnimalId = 2, 9, 50)))))) as petWeight
, iif(breed.AnimalId = 2, 1, 0) as catFlag 
, iif(breed.name like '%mix%', 1, 0) as mixedFlag
, iif(bf.HybridFlag = 1, 1, 0) as hybridFlag
, iif(s.countryId = 1, isnull(demog.IncomePerHouseHold, 59039), isnull(caninc.FSAmedianincome, 70336)) as income
, iif(s.countryId = 1, iif(isnumeric(demog.popDensity)=1, demog.popDensity, 1269.0), isnull(cad.PopDensityMi, 0.0)) as PopDensity
, iif(b.enrolldate >= '4/1/2017' AND b.NotPaidByCorpOrTruEmp = 0 AND b.samedaycancel = 0, 1, 0) as Enroll

from dwQuote.QuotePet qp with (nolock) 
inner join dwQuote.Quote q with (nolock) on (q.Id = qp.QuoteId)
inner join dwSalesLead.Person p with (nolock) on (p.Id = q.LeadId) 
inner join dwSalesLead.pet pp with (nolock) on (pp.PersonId = q.LeadId)
left join dw.State s with (nolock) on (s.Id = p.StateId)
left join dw.BreedV breed with (nolock) on (breed.id = pp.BreedId)
left join prc.BreedRollup br with (nolock) on (br.BreedId = breed.Id)
left join (select distinct * from sse.BreedFact) bf on (bf.BreedRollup = br.BreedRollup)
left join sse.DemographicsByZip demog with (nolock) on (demog.Zipcode = p.PostalCode)
left join sse.GLMCanadaFSAIncome caninc with (nolock) on caninc.FSA = left(p.PostalCode, 3)
left join sse.CanadianDemographics cad with (nolock) on cad.FSA = left(p.PostalCode, 3)
left join dw.Zipcode zc with (nolock) on (zc.Zipcode = p.PostalCode)
left join fct.enrTableV b with (nolock) on (b.PetId = pp.InsuredPetId)
where qp.CreatedOn >= '4/1/2017'
AND qp.CreatedOn < '5/1/2018'
AND p.platformid = 1
AND p.Id IS NOT NULL
AND ((b.EnrollDate >= '4/1/2017' AND b.[Path] = 'Web') OR b.EnrollDate IS NULL)

"""

# Call ODBC connection to Data Warehouse
engine = create_engine('mssql+pyodbc://sav-dwh1')
connection = engine.connect()

# Read query results into a pandas dataframe
df = pd.read_sql(query,connection)

connection.close()


# Inspect dataframe

# In[3]:


df.head()


# In[4]:


df['Enroll'].mean()


# In[5]:


df.dtypes


# In[6]:


df.shape


# In[7]:


## Create a training set and a testing set. Relative sizes to change later maybe.

train = df.sample(frac=0.85, random_state=123)
test = df.loc[~df.index.isin(train.index), :]


# In[8]:


## Break off the explanatory variables

x_train = train[['age', 'petWeight', 'catFlag', 'mixedFlag', 'hybridFlag', 'income', 'PopDensity']]
y_train = train['Enroll']
x_test = test[['age', 'petWeight', 'catFlag', 'mixedFlag', 'hybridFlag', 'income', 'PopDensity']]
y_test = test['Enroll']

x_train.head(3)


# In[9]:


x_test.shape


# In[10]:


# Resolving error on age field

np.any(np.isnan(x_train['age']))
np.all(np.isfinite(x_train['age']))


# Logit model below:

# In[66]:


# Training the takes very little time as it is not an ensemble method.

logit = linear_model.LogisticRegression(class_weight = 'balanced', random_state = 123)

logit.fit(x_train, y_train)


# In[23]:


preds = logit.predict_proba(x_train)


# In[31]:


test['pred']=logit.predict_proba(test.loc[:,('age', 'petWeight', 'catFlag', 'mixedFlag', 'hybridFlag', 'income', 'PopDensity')])


# In[67]:


X = x_train.iloc[:, 'age', 'income']
Y = y_train


# In[68]:


logit.fit(X, Y)


# In[69]:


x_min, x_max = X.iloc[:, 0].min() - .5, X.iloc[:, 0].max() + .5
y_min, y_max = X.iloc[:, 1].min() - .5, X.iloc[:, 1].max() + .5


# In[70]:


xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = logit.predict(np.c_[xx.ravel(), yy.ravel()])


# In[72]:


# Load results into color plot

Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(6,4))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot training points
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Pet Age')
plt.ylabel('Pet Weight')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13))
plt.yticks((10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120))

plt.show()


# In[76]:


# Load results into color plot

Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(6,4))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot training points
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Pet Age')
plt.ylabel('Pet Weight')

plt.xlim(0, 13.5)
plt.ylim(0, 160)
plt.xticks((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13))
plt.yticks((10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160))

plt.show()


# In[84]:


logit.fit(X, Y)


# In[85]:


X = x_train[['age', 'income']]
Y = y_train


# In[86]:


xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = logit.predict(np.c_[xx.ravel(), yy.ravel()])


# In[87]:


# Load results into color plot

Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(10,7))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot training points
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Pet Age')
plt.ylabel('Zip Median Income')

plt.xlim(0, 13.5)
plt.ylim(0, 200000)
plt.xticks((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13))
plt.yticks((10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000, 150000))

plt.show()


# In[96]:


10000*(1/3.)*.24


# In[ ]:


# 90% precision not silver bullet
# Need 24% phone conv
# Equations

