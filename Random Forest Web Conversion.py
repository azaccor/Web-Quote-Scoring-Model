
# coding: utf-8

# ## Predicting Conversions from Web Quotes

# In[1]:


import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import pyodbc
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import random


# Import the data from SQL Server

# In[2]:


query = """
select DISTINCT
	pp.Id as QuotePetLead
, isnull(iif(pp.AgeId = 33, 0, pp.AgeId), 3) as age
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
, qp.Premium
, iif(b.enrolldate >= '1/1/2018' AND b.NotPaidByCorpOrTruEmp = 0 AND b.samedaycancel = 0, 1, 0) as Enroll
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
where qp.CreatedOn >= '1/1/2018'
AND qp.CreatedOn < '5/1/2018'
AND p.platformid = 1
AND p.Id IS NOT NULL
AND ((b.EnrollDate >= '1/1/2018' AND b.[Path] = 'Web') OR b.EnrollDate IS NULL)

"""

# Call ODBC connection to Data Warehouse
engine = create_engine('mssql+pyodbc://sav-dwh1')
connection = engine.connect()

# Read query results into a pandas dataframe
df = pd.read_sql(query,connection)

connection.close()


# Inspect dataframe

# In[36]:


df.head()


# In[37]:


df['Enroll'].mean()


# In[38]:


df.dtypes


# In[39]:


df.shape


# In[40]:


## Create a training set and a testing set. Relative sizes to change later maybe.

train = df.sample(frac=0.85, random_state=123)
test = df.loc[~df.index.isin(train.index), :]


# In[41]:


## Break off the explanatory variables

x_train = train[['age', 'petWeight', 'catFlag', 'mixedFlag', 'hybridFlag', 'income', 'PopDensity', 'Premium']]
y_train = train['Enroll']
x_test = test[['age', 'petWeight', 'catFlag', 'mixedFlag', 'hybridFlag', 'income', 'PopDensity', 'Premium']]
y_test = test['Enroll']

x_train.head(3)


# In[42]:


x_test.shape


# In[43]:


# Resolving error on age field

np.any(np.isnan(x_train['age']))
np.all(np.isfinite(x_train['age']))


# No label encoding needed, no OneHotEncoding needed either. Nothing but numbers now.

# In[45]:


# Training the forest takes about 30 minutes on 1,000,000 observations growing 1,000 trees.

rfr = RandomForestRegressor(n_estimators=1000       # Number of trees
                             , criterion='mse'      # Metric of quality of split
                             , max_features=3        # How many features a single tree gets
                             , max_depth=None        # Max depth of the tree
                             , min_samples_split=2   # Min samples required to split node
                             , min_samples_leaf=10    # Min samples required to be in leaf node
                             , bootstrap=True       # Bootstrap the samples to build trees?
                             , oob_score=True        # OoB samples to estimate accuracy?
                             , random_state=123)
rfr.fit(x_train, y_train)


# In[46]:


print(rfr.feature_importances_)


# In[47]:


# Also time consuming

print(rfr.score(x_train, y_train))


# In[48]:


print(rfr.score(x_test, y_test))


# In[49]:


print(rfr.predict([[1, 110, 0, 0, 1, 95000, 3000, 120]]))


# In[50]:


preds = pd.DataFrame(rfr.predict(x_test))


# In[51]:


test['pred']=rfr.predict(test[['age', 'petWeight', 'catFlag', 'mixedFlag', 'hybridFlag', 'income', 'PopDensity', 'Premium']])


# In[52]:


test.to_csv("//Trushare/BI/Misc Projects/AZ_Misc_Files/Lead Lifecycle/PredictingWebEnrolls.csv", index = False)


# In[95]:


preds_df = x_test.join(preds)
preds_df = preds_df.join(y_test)


# In[96]:


preds_df.head()


# In[97]:


preds_df.to_csv("//Trushare/BI/Misc Projects/AZ_Misc_Files/Lead Lifecycle/preds.csv", index = False)


# In[ ]:


## rfc_dave = RandomForestClassifier(n_estimators=500
  #                           , criterion='entropy'
   #                          , max_depth=None
    #                         , min_samples_split=10
     #                        , min_samples_leaf=5
      #                       , min_weight_fraction_leaf=0.0
       #                      , max_features=0.2
        #                     , max_leaf_nodes=1000
         #                    , bootstrap=False
          #                   , oob_score=False
           #                  , n_jobs=1
            #                 , random_state=1234
             #                , verbose=0
              #               , warm_start=False
               #              , class_weight=None).fit(X_train_imputed, Y_train) 


# ### Recall that the categorical arguments must be entered in the following order:
# age (float), petWeight (int), catFlag (binary), mixedFlag (binary), hybridFlag(binary), income (int), and PopDensity (float).

# In[99]:


## Example1: 

print(rfc.predict_proba([[0.5, 40, 0, 0, 1, 95000, 3000]]))


