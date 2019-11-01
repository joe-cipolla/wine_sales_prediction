import xgboost as xgb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import operator
import os

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score



####################################  load & clean data  ###########################################
root_dir = os.path.abspath(os.curdir)
root_dir = '/'.join(root_dir.split('/')[:-1])  # nav up 1 directory from 'code' folder

source_path = root_dir + 'data/wine.csv'
source_df = pd.read_csv(source_path)
print(source_df.head(10))
print(source_df.describe())
print('Null values: ')
print(source_df.isnull().sum())
print(source_df.shape)

data = source_df.copy(deep=True)

data['TARGET_FLAG'] = np.array([data.TARGET > 0])[0].astype(int)
data['TARGET_AMT'] = data.TARGET - 1

M_STARS					= np.zeros(data.shape[0])
M_Density				= np.zeros(data.shape[0])
M_Sulphates				= np.zeros(data.shape[0])
M_Alcohol				= np.zeros(data.shape[0])
M_LabelAppeal			= np.zeros(data.shape[0])
M_TotalSulfurDioxide	= np.zeros(data.shape[0])
M_ResidualSugar			= np.zeros(data.shape[0])
M_Chlorides				= np.zeros(data.shape[0])
M_FreeSulfurDioxide		= np.zeros(data.shape[0])
M_pH					= np.zeros(data.shape[0])

M_STARS[data.STARS.isnull()]                            = 1
M_Density[data.Density.isnull()]                        = 1
M_Sulphates[data.Sulphates.isnull()]                    = 1
M_Alcohol[data.Alcohol.isnull()]                        = 1
M_LabelAppeal[data.LabelAppeal.isnull()]                = 1
M_TotalSulfurDioxide[data.TotalSulfurDioxide.isnull()]  = 1
M_ResidualSugar[data.ResidualSugar.isnull()]            = 1
M_Chlorides[data.Chlorides.isnull()]                    = 1
M_FreeSulfurDioxide[data.FreeSulfurDioxide.isnull()]    = 1
M_pH[data.pH.isnull()]                                  = 1

data['IMP_STARS']               = data['STARS'].fillna(0)
data['IMP_Density']             = data['Density'].fillna(0.9942027)
data['IMP_Sulphates']           = data['Sulphates'].fillna(0.5271118)
data['IMP_Alcohol']             = data['Alcohol'].fillna(10.4892363)
data['IMP_LabelAppeal']         = data['LabelAppeal'].fillna(-0.009066)
data['IMP_TotalSulfurDioxide']  = data['TotalSulfurDioxide'].fillna(120.7142326)
data['IMP_ResidualSugar']       = data['ResidualSugar'].fillna(5.4187331)
data['IMP_Chlorides']           = data['Chlorides'].fillna(0.0548225)
data['IMP_FreeSulfurDioxide']   = data['FreeSulfurDioxide'].fillna(30.8455713)
data['IMP_pH']                  = data['pH'].fillna(3.2076282)

data['Norm_FixedAcidity']		= 	(data.FixedAcidity - 7.0757171) / 6.3176435
data['Norm_VolatileAcidity'] 	= 	(data.VolatileAcidity - 0.3241039) / 0.7840142
data['Norm_CitricAcid'] 		= 	(data.CitricAcid - 0.3084127) / 0.8620798
data['Norm_ResidualSugar'] 		= 	(data.IMP_ResidualSugar - 5.4187331) / 33.7493790
data['Norm_Chlorides'] 			= 	(data.IMP_Chlorides - 0.0548225) / 0.3184673
data['Norm_FreeSulfurDioxide'] 	= 	(data.IMP_FreeSulfurDioxide - 30.8455713) / 148.7145577
data['Norm_TotalSulfurDioxide'] = 	(data.IMP_TotalSulfurDioxide - 120.7142326) / 231.9132105
data['Norm_Density'] 			= 	(data.Density - 0.9942027) / 0.0265376
data['Norm_pH'] 				= 	(data.IMP_pH - 3.2076282) / 0.6796871
data['Norm_Sulphates']			= 	(data.IMP_Sulphates - 0.5271118) / 0.9321293
data['Norm_Alcohol']			= 	(data.IMP_Alcohol - 10.4892363) / 3.7278190
data['Norm_AcidIndex']			= 	(data.AcidIndex - 7.7727237) / 1.3239264


data = data.drop(columns=['INDEX', 'ResidualSugar', 'Chlorides', 'FreeSulfurDioxide',
                          'TotalSulfurDioxide', 'pH', 'Sulphates', 'Alcohol', 'STARS'])

cols = ['TARGET', 'TARGET_AMT', 'TARGET_FLAG', 'FixedAcidity', 'VolatileAcidity', 'CitricAcid',
        'Density', 'LabelAppeal', 'AcidIndex', 'IMP_STARS', 'IMP_Density', 'IMP_Sulphates',
        'IMP_Alcohol', 'IMP_LabelAppeal', 'IMP_TotalSulfurDioxide', 'IMP_ResidualSugar',
        'IMP_Chlorides', 'IMP_FreeSulfurDioxide', 'IMP_pH', 'Norm_FixedAcidity',
        'Norm_VolatileAcidity', 'Norm_CitricAcid', 'Norm_ResidualSugar', 'Norm_Chlorides',
        'Norm_FreeSulfurDioxide', 'Norm_TotalSulfurDioxide', 'Norm_Density', 'Norm_pH',
        'Norm_Sulphates', 'Norm_Alcohol', 'Norm_AcidIndex']

data = data[cols]

buyer_df = data[data.TARGET > 0]



##### setup TEST FILE
test_path = '/Users/joecipolla/Documents/Reference/Education/411-DL_Predictive Modeling II/' \
              'Unit_03__Poisson_Regression/Wine_Sales_Problem/wine_test.csv'
source_test_df = pd.read_csv(test_path)

test = source_test_df.copy(deep=True)

test['TARGET_FLAG'] = np.array([test.TARGET > 0])[0].astype(int)
test['TARGET_AMT'] = test.TARGET - 1

M_STARS					= np.zeros(test.shape[0])
M_Density				= np.zeros(test.shape[0])
M_Sulphates				= np.zeros(test.shape[0])
M_Alcohol				= np.zeros(test.shape[0])
M_LabelAppeal			= np.zeros(test.shape[0])
M_TotalSulfurDioxide	= np.zeros(test.shape[0])
M_ResidualSugar			= np.zeros(test.shape[0])
M_Chlorides				= np.zeros(test.shape[0])
M_FreeSulfurDioxide		= np.zeros(test.shape[0])
M_pH					= np.zeros(test.shape[0])

M_STARS[test.STARS.isnull()]                            = 1
M_Density[test.Density.isnull()]                        = 1
M_Sulphates[test.Sulphates.isnull()]                    = 1
M_Alcohol[test.Alcohol.isnull()]                        = 1
M_LabelAppeal[test.LabelAppeal.isnull()]                = 1
M_TotalSulfurDioxide[test.TotalSulfurDioxide.isnull()]  = 1
M_ResidualSugar[test.ResidualSugar.isnull()]            = 1
M_Chlorides[test.Chlorides.isnull()]                    = 1
M_FreeSulfurDioxide[test.FreeSulfurDioxide.isnull()]    = 1
M_pH[test.pH.isnull()]                                  = 1

test['IMP_STARS']               = test['STARS'].fillna(0)
test['IMP_Density']             = test['Density'].fillna(0.9942027)
test['IMP_Sulphates']           = test['Sulphates'].fillna(0.5271118)
test['IMP_Alcohol']             = test['Alcohol'].fillna(10.4892363)
test['IMP_LabelAppeal']         = test['LabelAppeal'].fillna(-0.009066)
test['IMP_TotalSulfurDioxide']  = test['TotalSulfurDioxide'].fillna(120.7142326)
test['IMP_ResidualSugar']       = test['ResidualSugar'].fillna(5.4187331)
test['IMP_Chlorides']           = test['Chlorides'].fillna(0.0548225)
test['IMP_FreeSulfurDioxide']   = test['FreeSulfurDioxide'].fillna(30.8455713)
test['IMP_pH']                  = test['pH'].fillna(3.2076282)

test['Norm_FixedAcidity']		= 	(test.FixedAcidity - 7.0757171) / 6.3176435
test['Norm_VolatileAcidity'] 	= 	(test.VolatileAcidity - 0.3241039) / 0.7840142
test['Norm_CitricAcid'] 		= 	(test.CitricAcid - 0.3084127) / 0.8620798
test['Norm_ResidualSugar'] 		= 	(test.IMP_ResidualSugar - 5.4187331) / 33.7493790
test['Norm_Chlorides'] 			= 	(test.IMP_Chlorides - 0.0548225) / 0.3184673
test['Norm_FreeSulfurDioxide'] 	= 	(test.IMP_FreeSulfurDioxide - 30.8455713) / 148.7145577
test['Norm_TotalSulfurDioxide'] = 	(test.IMP_TotalSulfurDioxide - 120.7142326) / 231.9132105
test['Norm_Density'] 			= 	(test.Density - 0.9942027) / 0.0265376
test['Norm_pH'] 				= 	(test.IMP_pH - 3.2076282) / 0.6796871
test['Norm_Sulphates']			= 	(test.IMP_Sulphates - 0.5271118) / 0.9321293
test['Norm_Alcohol']			= 	(test.IMP_Alcohol - 10.4892363) / 3.7278190
test['Norm_AcidIndex']			= 	(test.AcidIndex - 7.7727237) / 1.3239264


test = test.drop(columns=['INDEX', 'ResidualSugar', 'Chlorides', 'FreeSulfurDioxide',
                          'TotalSulfurDioxide', 'pH', 'Sulphates', 'Alcohol', 'STARS'])

cols = ['TARGET', 'TARGET_AMT', 'TARGET_FLAG', 'FixedAcidity', 'VolatileAcidity', 'CitricAcid',
        'Density', 'LabelAppeal', 'AcidIndex', 'IMP_STARS', 'IMP_Density', 'IMP_Sulphates',
        'IMP_Alcohol', 'IMP_LabelAppeal', 'IMP_TotalSulfurDioxide', 'IMP_ResidualSugar',
        'IMP_Chlorides', 'IMP_FreeSulfurDioxide', 'IMP_pH', 'Norm_FixedAcidity',
        'Norm_VolatileAcidity', 'Norm_CitricAcid', 'Norm_ResidualSugar', 'Norm_Chlorides',
        'Norm_FreeSulfurDioxide', 'Norm_TotalSulfurDioxide', 'Norm_Density', 'Norm_pH',
        'Norm_Sulphates', 'Norm_Alcohol', 'Norm_AcidIndex']

test = test[cols]



###########################################  EDA  ##################################################



base_pal = ['#625EC1']   ##7556FF
flatui = ['#bfd3e6', '#a9c4de', '#98b0d3', '#8c95c6', '#8c79b8', '#8b5daa', '#88409c',
          '#831f86', '#6f0a6b']
sns.set()
sns.set_context("talk")
sns.set_palette(flatui)
# sns.countplot(x=data['TARGET'].dropna().astype(int), palette=sns.color_palette(flatui))
sns.countplot(x=data['TARGET'].dropna().astype(int), palette=base_pal)
plt.title('Number of Cases Purchased')
plt.xlabel('TARGET')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('Histogram__TARGET')
plt.show()

sns.set_palette(base_pal)
sns.distplot(data['TARGET'].dropna().astype(int))
plt.title('Number of Cases Purchased')
plt.xlabel('TARGET')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('Histogram__TARGET2')
plt.show()

sns.countplot(data['AcidIndex'].dropna().astype(int), hue=data['TARGET_FLAG'],
              palette=sns.set_palette(['#98b0d3', '#8b5daa']))
plt.title('Acid Index vs. TARGET_FLAG')
plt.xlabel('AcidIndex')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('Histogram__AcidIndex_byFlag')
plt.show()

sns.countplot(source_df['STARS'].dropna().astype(int), hue=source_df['TARGET'],
              palette=flatui)
plt.title('STARS vs. TARGET')
plt.xlabel('STARS')
plt.ylabel('Count')
plt.tight_layout()
plt.legend(loc='upper right')
plt.savefig('Histogram__STARS')
plt.show()

sns.set_palette(base_pal)
f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=False)
plt.suptitle('Chemical Variable Distributions')
sns.distplot(data['FixedAcidity'], ax=axes[0, 0])
sns.distplot(data['CitricAcid'], ax=axes[0, 1])
sns.distplot(data['IMP_ResidualSugar'], ax=axes[1, 0])
sns.distplot(data['IMP_Chlorides'], ax=axes[1, 1])
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('Distributions__Multi')
plt.show()

sns.set_palette(flatui)
sns.violinplot(data['TARGET_FLAG'], data['IMP_STARS'], palette=sns.color_palette(flatui))
plt.title('Sales by STARS')
plt.xlabel('TARGET_FLAG')
plt.ylabel('IMP_STARS')
plt.tight_layout()
plt.savefig('Viloin__TARGET_FLAG_BY_STARS')
plt.show()

sns.violinplot(data['TARGET_FLAG'], data['FixedAcidity'], palette=sns.color_palette(flatui))
plt.title('Sales by FixedAcidity')
plt.xlabel('TARGET_FLAG')
plt.ylabel('FixedAcidity')
plt.tight_layout()
plt.savefig('Viloin__TARGET_FLAG_BY_FixedAcidity')
plt.show()


sns.violinplot(data['TARGET_FLAG'], data['IMP_LabelAppeal'], palette=sns.color_palette(flatui))
plt.title('Sales by LabelAppeal')
plt.xlabel('TARGET_FLAG')
plt.ylabel('IMP_LabelAppeal')
plt.tight_layout()
plt.savefig('Viloin__TARGET_FLAG_BY_LabelAppeal')
plt.show()

sns.violinplot(data['TARGET'], data['IMP_LabelAppeal'], palette=sns.color_palette(flatui))
plt.title('Sales by LabelAppeal')
plt.xlabel('TARGET')
plt.ylabel('IMP_LabelAppeal')
plt.tight_layout()
plt.savefig('Viloin__TARGET_BY_LabelAppeal')
plt.show()

sns.violinplot(data['TARGET'], data['IMP_STARS'], palette=sns.color_palette(flatui))
plt.title('Sales by STARS')
plt.xlabel('TARGET')
plt.ylabel('IMP_STARS')
plt.tight_layout()
plt.savefig('Viloin__TARGET_BY_STARS')
plt.show()

purps = sns.cubehelix_palette(8, start=2.8, rot=0.1)
# purps = sns.cubehelix_palette()
# sns.choose_colorbrewer_palette(data_type='qualitative', as_cmap=False)
df = pd.DataFrame(data['TARGET']).join(data[['FixedAcidity', 'CitricAcid', 'IMP_ResidualSugar']])
sns.pairplot(df.sample(500), kind='scatter', hue='TARGET', palette=flatui)
plt.savefig('pairplot_acid_sugar_chlor.png')
plt.show()

df = pd.DataFrame(data['TARGET']).join(data[['VolatileAcidity', 'LabelAppeal', 'Norm_Alcohol']])
sns.pairplot(df.sample(500), kind='scatter', hue='TARGET', palette=flatui)
plt.savefig('pairplot_acid_label_ph.png')
plt.show()

df = pd.DataFrame(data['TARGET']).join(data[['IMP_STARS', 'LabelAppeal', 'AcidIndex']])
sns.pairplot(df.sample(500), kind='scatter', hue='TARGET', palette=flatui)
plt.savefig('pairplot_stars_label_acid.png')
plt.show()

#########################################  Modeling  ###############################################


### PROC REG  -- stepwise selection  -- predicting TARGET

### PROC GENMOD  --  using stepwise selected variables  --  log link dist=nb

### PROC Logisitc  --  using stepwise selected variables  --




########################################  XGBoost  ###############################################
X = data.iloc[:, 3:]
y = data.iloc[:, 0]
dTrain = xgb.DMatrix(data=X, label=y)
params = {'objective': 'count:poisson', 'max_depth': 4}
cv_results = xgb.cv(dtrain=dTrain, params=params, nfold=4, num_boost_round=10, metrics='error',
                    as_pandas=True)
print("Accuracy: %f" % ((1 - cv_results["test-error-mean"]).iloc[-1]))
bst = xgb.train(params, dTrain)
preds = bst.predict(dTrain)
print("RMSE: %f" % np.sqrt(mean_squared_error(y, preds)))


X = data.iloc[:, 3:]
y = data.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
xg_reg = xgb.XGBRegressor(objective='reg:linear', n_estimators=10, seed=123)
xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % rmse)

DM_train = xgb.DMatrix(data=X_train, label=y_train)
DM_test = xgb.DMatrix(data=X_test, label=y_test)
params = {"booster": "gblinear", "objective": "reg:linear"}
xg_reg = xgb.train(params=params, dtrain=DM_train, num_boost_round=10)
preds = xg_reg.predict(DM_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % rmse)

dmatrix = xgb.DMatrix(data=X, label=y)
params = {"objective": "reg:linear", "max_depth": 4}
l1_params = [1, 10, 100]
rmses_l1 = []
for reg in l1_params:
    params["alpha"] = reg
    cv_results = xgb.cv(dtrain=dmatrix, params=params, nfold=4, num_boost_round=10,
                        metrics="rmse", as_pandas=True, seed=123)
    rmses_l1.append(cv_results["test-rmse-mean"].tail(1).values[0])

print("Best rmse as a function of l1:")
print(pd.DataFrame(list(zip(l1_params, rmses_l1)), columns=["l1", "rmse"]))

dmatrix = xgb.DMatrix(data=X, label=y)
untuned_params = {"objective": "reg:linear"}
untuned_cv_results_rmse = xgb.cv(dtrain=dmatrix, params=untuned_params, nfold=4, metrics="rmse",
                                 as_pandas=True, seed=123)
print("Untuned rmse: %f" % ((untuned_cv_results_rmse["test-rmse-mean"]).tail(1).values[0]))

tuned_params = {"objective": "reg:linear", 'colsample_bytree': 0.3, 'learning_rate': 0.1,
                'max_depth': 5}
tuned_cv_results_rmse = xgb.cv(dtrain=dmatrix, params=tuned_params, nfold=4,
                               num_boost_round=200, metrics="rmse", as_pandas=True, seed=123)
print("Tuned rmse: %f" % ((tuned_cv_results_rmse["test-rmse-mean"]).tail(1).values[0]))


dmatrix = xgb.DMatrix(data=X, label=y)
gbm_param_grid = {'learning_rate': [0.1],
                  'n_estimators': [100],
                  'subsample': [0.9],
                  'max_depth': [3]}
gbm = xgb.XGBRegressor()
grid_mse = GridSearchCV(estimator=gbm,
                        param_grid=gbm_param_grid,
                        scoring='neg_mean_squared_error', cv=4, verbose=1)
grid_mse.fit(X, y)
print("Best parameters found: ", grid_mse.best_params_)
print("Lowest Grid Search RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))
# Best parameters found:  {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.9}
# Lowest Grid Search RMSE found:  1.2367619901527913

randomized_mse = RandomizedSearchCV(estimator=gbm,
                                    param_distributions=gbm_param_grid,
                                    scoring='neg_mean_squared_error',
                                    cv=4, verbose=1)
randomized_mse.fit(X, y)
print("Best paramters found: ", randomized_mse.best_params_)
print("Lowest Randomized Search RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))


# features = X.columns.tolist()
clr = xgb.train(params=grid_mse.best_params_, dtrain=DM_train)
xgb.plot_importance(clr)
plt.title('XGBoost Feature Importance')
plt.xlabel('Relative Importance')
plt.ylabel('')
plt.tight_layout()
plt.gcf().savefig('feature_importance_xgb.png')
plt.show()

dot = xgb.to_graphviz(clr, num_trees=2)
dot.render('graphviz_2.gv.png', view=True)

# xgb.plot_tree(clr)
# fig = plt.gcf()
# fig.set_size_inches(150, 100)
# fig.savefig('tree.png')

### create scoring csv
gbm = xgb.XGBRegressor(**grid_mse.best_params_)
gbm.fit(X, y)

X_test = test.iloc[:, 3:]

preds = gbm.predict(X_test)
preds = pd.DataFrame(data=preds)
preds[preds < 0] = 0
preds = preds.round(0).astype(int)
idx = pd.DataFrame(source_test_df.INDEX)
preds = idx.join(preds)
preds.to_csv('XGBRegressor.csv')




#
# randomized_mse = RandomizedSearchCV(estimator=gbm, param_distributions=gbm_param_grid, n_iter=12,
#                                     scoring='neg_mean_squared_error', cv=4, verbose=1)
# randomized_mse.fit(X, y)
# print("Best parameters found: ", randomized_mse.best_params_)
# print("Lowest Randomized Search RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))
#

# rf_pipeline = Pipeline[("st_scaler", StandardScaler()), ("rf_model", RandomForestRegressor())]
# scores = cross_val_score(rf_pipeline, X, y, scoring="neg_mean_squared_error", cv=10)
# final_avg_rmse = np.mean(np.sqrt(np.abs(scores)))
# print("Final RMSE:", final_avg_rmse)





### SCORE  and compare all modeling results
