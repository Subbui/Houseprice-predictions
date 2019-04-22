# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


#Append train and test data

train_id = train['Id']
test_id = test['Id']
train.drop('Id',axis =1, inplace=True)
test.drop('Id',axis=1, inplace=True)
train_obs=train.shape[0]
test_obs=test.shape[0]
train_y = train.SalePrice.values
data=pd.concat((train,test)).reset_index(drop=True)
data.drop(['SalePrice'],axis=1,inplace=True)

# Lets repalce GarageYrBlt with 0 as if the house has no garage then it make sense to replace null values with 0. similar is the case with GarageCars and GaragaeArea
data['GarageYrBlt'] = data['GarageYrBlt'].fillna(0)
data['GarageArea'] = data['GarageArea'].fillna(0)
data['GarageCars'] = data['GarageCars'].fillna(0)

#MasVnrArea can be replaced with 0 as there is no masonry veneer for those houses

data['MasVnrArea'] = data['MasVnrArea'].fillna(0)

#LotFrontage has 486 null values. Fill the missing values by the median of the neighbor
data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

        
#Variable Alley has 2721 'NaN' values, which means they don't have alley access. similar is the case with MiscFeature PookQC FireplaceQu
data['Alley'] = data['Alley'].fillna('None')
data['MiscFeature'] = data['MiscFeature'].fillna('None')
data['PoolQC'] = data['PoolQC'].fillna('None')
data['FireplaceQu']= data['FireplaceQu'].fillna('None')
data['BsmtCond'] = data['BsmtCond'].fillna('None')
data['BsmtExposure'] = data['BsmtExposure'].fillna('None')
data['BsmtFinType1'] = data['BsmtFinType1'].fillna('None')
data['BsmtFinType2'] = data['BsmtFinType2'].fillna('None')
data['BsmtQual'] = data['BsmtQual'].fillna('None')
data['Fence'] = data['Fence'].fillna('None')
data['GarageCond'] = data['GarageCond'].fillna('None')
data['GarageFinish'] = data['GarageFinish'].fillna('None')
data['GarageQual'] = data['GarageQual'].fillna('None')
data['GarageType'] = data['GarageType'].fillna('None')
data['MasVnrType'] = data['MasVnrType'].fillna('None')

#Since only 1 missing, i will be replacing it with 0
data['Electrical'] = data['Electrical'].fillna(0)
data['Exterior1st'] = data['Exterior1st'].fillna(0)
data['Exterior2nd'] = data['Exterior2nd'].fillna(0)

for col in ('BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtHalfBath','BsmtUnfSF','TotalBsmtSF'):
    data[col] = data[col].fillna(0)    



data['Functional'] = data['Functional'].fillna('Typ')
data['KitchenQual'] = data['KitchenQual'].fillna('TA')
data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode()[0])
data['SaleType'] =  data['SaleType'].fillna(data['SaleType'].mode()[0])

#Since the variable is not giving us any info
data= data.drop(['Utilities'],axis=1)

#Transofrm numerical variables into categorical ones
data['YrSold'] = data['YrSold'].astype( str)
data['YearRemodAdd'] = data['YearRemodAdd'].astype(str)
data['YearBuilt'] = data['YearBuilt'].astype(str)
data['OverallQual'] = data['OverallQual'].astype(str)
data['OverallCond'] = data['OverallCond'].astype(str)
data['MoSold'] = data['MoSold'].astype(str)
data['MSSubClass'] = data['MSSubClass'].astype(str)

data=pd.get_dummies(data)

train = data[:train_obs]
test = data[train_obs:]

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

SEED = 1512
rf = RandomForestRegressor(n_estimators=128, random_state=SEED)
param_grid = [
    {
    'max_depth': [2, 4, 8, 16, 32, 64],
    'min_samples_split': [2, 4, 8, 16, 32],
    'min_samples_leaf': [2, 4, 8, 16],
    }
]

grid_search = GridSearchCV(rf, param_grid, cv=5,  n_jobs=24, verbose=1)

rf_model = grid_search.fit(train,train_y)
rf_model.best_score_
rf_model.best_params_

test_y = rf_model.predict(test)

submission = pd.DataFrame({'Id' : test_id, 'SalePrice' : test_y})
submission.to_csv('submission.csv',index = False)