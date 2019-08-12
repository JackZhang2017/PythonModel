# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 15:56:12 2018


"""


import numpy as np
import pandas as pd
from pandas import DataFrame,Series
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import lightgbm as lgb
from sklearn import metrics
from sklearn.model_selection import GridSearchCV 

t_fea=pd.read_csv('flightticket.csv',sep='	')
city_fea = pd.read_csv('city_score_june_19.csv',sep=',')
flight_num = pd.read_csv('flight_num.csv',sep=',')
order_city=pd.read_csv('order_city.csv',sep=',')
search_city=pd.read_csv('search_city.csv',sep=',')

#数据预处理
t_fea.replace('-',np.nan,inplace=True)

city_fea.drop_duplicates(['city'],inplace=True)
flight_num.drop_duplicates(['flight_num'],inplace=True)
#order_city_fea.drop_duplicates(['order_city_feature.arr_city'],inplace=True)



city_col = ['city', 'no_hot', 'grassland', 'cityscape', 'geology', 'wildlife_park', 'shopping', 'sea', 'red_tourism', 'building',
       'river_sand_lakes', 'history', 'food', 'folk_custom', 'forest', 'desert', 'mountain', 'flowers', 'world_heritage', 'hydrology', 'astronomy',
       'art', 'amusement_park', 'gardens', 'religion', 'natural_landsacpe', 'entertainment', 'island', 'arr_country']
city_fea.columns = city_col 

#去掉空值超过50%的数据，暂时不删除
def qu_na(df):
    col_name = []
    for col in list(df.columns):
        if df[col].isnull().sum() / df[col].size > 0.9:
            col_name.append(col)
    df.drop(col_name,axis=1,inplace=True)
    return df
        
qu_na(t_fea)

#填充第一次
t_fea.fillna(-1,inplace=True)

#日期特征处理
print ('start dealing with date labels...')

date_label = ['fact_feature_label.user_trip_date_low',
       'fact_feature_label.user_trip_date_high',
       'fact_feature_label.go_date']

#def goin(df):
#    godate_isin_lowhigh = []
#    for index in df.index:
#        if (t_fea[date_label[2]][index] != -1) & (t_fea[date_label[0]][index] != -1) & (t_fea[date_label[1]][index] != -1):
#            entime = datetime.strptime(t_fea[date_label[2]][index],'%Y-%m-%d')
#            entime_low = datetime.strptime(t_fea[date_label[0]][index],'%Y-%m-%d')
#            entime_high = datetime.strptime(t_fea[date_label[1]][index],'%Y-%m-%d')
#            if (entime-entime_low).days>=0 and (entime_high-entime).days>=0:
#                godate_isin_lowhigh.append(1)
#            else:
#                godate_isin_lowhigh.append(0)
#        else:
#            godate_isin_lowhigh.append(-1)
#    df['godate_isin_lowhigh'] = godate_isin_lowhigh

def goin(df):
    godate_isin_lowhigh = []
    for index in df.index:
        if (t_fea[date_label[2]][index] != -1) & (t_fea[date_label[0]][index] != -1) & (t_fea[date_label[1]][index] != -1):
            entime = datetime.strptime(t_fea[date_label[2]][index],'%Y-%m-%d')
            entime_low = datetime.strptime(t_fea[date_label[0]][index],'%Y-%m-%d')
            entime_high = datetime.strptime(t_fea[date_label[1]][index],'%Y-%m-%d')
            if (entime-entime_low).days>=0 and (entime_high-entime).days>=0:
                godate_isin_lowhigh.append(0)
            else:
                dh = abs((entime-entime_low).days)
                godate_isin_lowhigh.append(dh)
        else:
            godate_isin_lowhigh.append(-1)
    df['godate_isin_lowhigh'] = godate_isin_lowhigh

goin(t_fea)
print ('goin over')


trip_dic = locals()
for j in date_label:      #[12,13,21]    12全是- [14,15,24,25,27]
    ww = []
    for index in t_fea.index:
        if t_fea[j][index] != -1:
            entime = datetime.strptime(t_fea[j][index],'%Y-%m-%d')
            week = entime.weekday() + 1
            ww.append(week)
        else:
            ww.append(-1)
    trip_dic['W_%s' %j] = ww
    t_fea['W_%s' %j] = trip_dic['W_%s' %j]
print('trip_date_week over')


def lianjie(df1,df2,df3,df4,df5):
    df2.rename(columns={'city':'fact_feature_label.arrcity'},inplace=True)
    df3.rename(columns={'flight_num':'fact_feature_label.flight_no'},inplace=True)
    df4.rename(columns={'t2.arr_city':'fact_feature_label.arrcity','t2.dom_inter':'dom_inter','dt':'fact_feature_label.dt'},inplace=True)
    df5.rename(columns={'t1.arr_city':'fact_feature_label.arrcity','t1.dom_inter':'dom_inter','dt':'fact_feature_label.dt'},inplace=True)
    df1 = pd.merge(df1,df2,how = 'left',on='fact_feature_label.arrcity')
    df1 = pd.merge(df1,df3,how = 'left',on='fact_feature_label.flight_no')
    df1 = pd.merge(df1,df4,how = 'left',on=['fact_feature_label.arrcity','fact_feature_label.dt'])
    df1 = pd.merge(df1,df5,how = 'left',on=['fact_feature_label.arrcity','dom_inter','fact_feature_label.dt'])
    return df1

t_fea = lianjie(t_fea,city_fea,flight_num,order_city,search_city)
print('merge over')

#数据2处理
t_fea.fillna(-1,inplace=True)
print('fill over')



# 将城市文字转label
word_label = [ 'fact_feature_label.gender',
       'fact_feature_label.permanent_place',
       'fact_feature_label.flight_no', 
       'fact_feature_label.depcity',
       'fact_feature_label.arrcity',
       'arr_country',
       'airline_company'
       ]

label_encoder = LabelEncoder()
def strtonum(df):
    for i in word_label:   #[0,1,3,7,16,17,18],[0,1,3,8,19,20,21]
        values = np.array(df[i],dtype='str')
        integer_encoded = label_encoder.fit_transform(values)
        df[i] = integer_encoded
        df[i] = df[i].astype('category') 
    return df

strtonum(t_fea)
print('word_label transform over')


#年龄分段,作用不大，反而抑制。
def fenduan(df):
    age = pd.cut(df['fact_feature_label.age'],4)
    age_encoded = label_encoder.fit_transform(age)
    df['fact_feature_label.age_fen'] = age_encoded
    df['fact_feature_label.age_fen'] = df['fact_feature_label.age_fen'].astype('category') 
    return df

fenduan(t_fea)
print('cut over')

print('feature deal over')



#测试集
columns1 =['fact_feature_label.age',
           'fact_feature_label.gender',
       'fact_feature_label.price_sensitive',
       'fact_feature_label.is_business_user',
       'fact_feature_label.permanent_place',
       'fact_feature_label.is_student_user',
       'fact_feature_label.no_weekend_times', 
       'fact_feature_label.inter_num',
       'fact_feature_label.travel_days',
#       'fact_feature_label.user_trip_date_low',
#       'fact_feature_label.user_trip_date_high',
       'fact_feature_label.weekend_times',
       'fact_feature_label.user_search_city_num_month',
       'fact_feature_label.flight_no',
       'fact_feature_label.depcity',
       'fact_feature_label.arrcity', 
       'fact_feature_label.price',
       'fact_feature_label.flight_type', 
#       'fact_feature_label.go_date',
#       'fact_feature_label.click',
#       'fact_feature_label.dt', 
       'godate_isin_lowhigh',
       'W_fact_feature_label.user_trip_date_low',
       'W_fact_feature_label.user_trip_date_high',
       'W_fact_feature_label.go_date',
       'no_hot', 
       'grassland',
       'cityscape',
       'geology', 
       'wildlife_park',
       'shopping',
       'sea',
       'red_tourism', 
       'building',
       'river_sand_lakes',
       'history',
       'food', 
       'folk_custom',
       'forest',
       'desert',
       'mountain',
       'flowers',
       'world_heritage',
       'hydrology', 
       'astronomy', 
       'art',
       'amusement_park', 
       'gardens', 
       'religion', 
       'natural_landsacpe',
       'entertainment',
       'island',
       'arr_country',
       'airline_company',
       'airline_transfer_times',
       'flight_order_times', 
       'flight_order_times_ratio ',
       'fact_feature_label.age_fen',
        'dom_inter', 
        't2.city_order_times',
       'city_order_times_ratio',
       't1.city_search_times',
       'city_search_times_ratio'
         ]
#
#
category_f = [
#        'fact_feature_label.uid', 
#        'fact_feature_label.qunar_username',
#       'fact_feature_label.gender',
       'fact_feature_label.is_business_user',
       'fact_feature_label.is_student_user',
#       'fact_feature_label.permanent_place',
#       'fact_feature_label.flight_no', 
#       'fact_feature_label.depcity',
#       'fact_feature_label.arrcity', 
#       'fact_feature_label.flight_type', 
#       'godate_isin_lowhigh',
#       'W_fact_feature_label.user_trip_date_low',
#       'W_fact_feature_label.user_trip_date_high',
#       'W_fact_feature_label.go_date',
#       'no_hot', 
#       'grassland',
#       'cityscape',
#       'geology', 
#       'wildlife_park',
#       'shopping',
#       'sea',
#       'red_tourism', 
#       'building',
#       'river_sand_lakes',
#       'history',
#       'food', 
#       'folk_custom',
#       'forest',
#       'desert',
#       'mountain',
#       'flowers',
#       'world_heritage',
#       'hydrology', 
#       'astronomy', 
#       'art',
#       'amusement_park', 
#       'gardens', 
#       'religion', 
#       'natural_landsacpe',
#       'entertainment',
#       'island',
#       'arr_country',
#       'airline_company',
#       'fact_feature_label.age_fen'
        ]

#X = result[columns1]
#y = result['y']
#
#X_for_test = result_test[columns1]
#Y_for_test = result_test['y']


#from sklearn.cross_validation import train_test_split
#t_fea['fact_feature_label.click'] = t_fea['fact_feature_label.click'].astype(int)

#X_for_test = t_fea[t_fea['fact_feature_label.dt'] == '2018-06-18'][columns1]
#Y_for_test = t_fea[t_fea['fact_feature_label.dt'] == '2018-06-18']['fact_feature_label.click']
def leixing(df):
    for i in category_f:
        df[i] = df[i].astype('category') 
    return df
leixing(t_fea)

print('category over...')

X_train = t_fea[(t_fea['fact_feature_label.dt'] == '2018-06-15')|(t_fea['fact_feature_label.dt'] == '2018-06-16')][columns1]
y_train = t_fea[(t_fea['fact_feature_label.dt'] == '2018-06-15')|(t_fea['fact_feature_label.dt'] == '2018-06-16')]['fact_feature_label.click']

X_test = t_fea[t_fea['fact_feature_label.dt'] == '2018-06-17'][columns1]
y_test = t_fea[t_fea['fact_feature_label.dt'] == '2018-06-17']['fact_feature_label.click']


lgb_train = lgb.Dataset(X_train, y_train)  #feature_name=columns1,
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, categorical_feature=category_f)
#lgb_for_test = lgb.Dataset(X_for_test)

### 设置初始参数--不含交叉验证参数
print('设置参数')
params = {
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric': 'binary_logloss',
          }

### 交叉验证(调参)
print('交叉验证')
max_auc = float('Inf')
best_params = {}

# 准确率
print("调参1：提高AUC")
for num_leaves in range(20,200,5):
    for max_depth in range(3,8,1):
        params['num_leaves'] = num_leaves
        params['max_depth'] = max_depth
        cv_results = lgb.cv(
                            params,
                            lgb_train,
                            seed=42,
                            nfold=3,
                            metrics=['auc'],
                            early_stopping_rounds=10,
                            verbose_eval=True
                            )
        mean_auc = pd.Series(cv_results['auc-mean']).max()
        boost_rounds = pd.Series(cv_results['auc-mean']).argmax()
        if mean_auc < max_auc:
            max_auc = mean_auc
            best_params['num_leaves'] = num_leaves
            best_params['max_depth'] = max_depth

params['num_leaves'] = best_params['num_leaves']
params['max_depth'] = best_params['max_depth']

# 过拟合
print("调参2：降低过拟合")
for max_bin in range(1,255,5):
    for min_sum_hessian_in_leaf in range(10,200,5):
        params['max_bin'] = max_bin
        params['min_sum_hessian_in_leaf'] = min_sum_hessian_in_leaf
        cv_results = lgb.cv(
                            params,
                            lgb_train,
                            seed=42,
                            nfold=3,
                            metrics=['auc'],
                            early_stopping_rounds=3,
                            verbose_eval=True
                            )
        mean_auc = pd.Series(cv_results['auc-mean']).max()
        boost_rounds = pd.Series(cv_results['auc-mean']).argmax()
        if mean_auc < max_auc:
            max_auc = mean_auc
            best_params['max_bin']= max_bin
            best_params['min_sum_hessian_in_leaf'] = min_sum_hessian_in_leaf

params['min_sum_hessian_in_leaf'] = best_params['min_sum_hessian_in_leaf']
params['max_bin'] = best_params['max_bin']

print("调参3：降低过拟合")
for bagging_fraction in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    for bagging_freq in range(0,50,5):
        params['bagging_fraction'] = bagging_fraction
        params['bagging_freq'] = bagging_freq
        cv_results = lgb.cv(
                            params,
                            lgb_train,
                            seed=42,
                            nfold=3,
                            metrics=['auc'],
                            early_stopping_rounds=3,
                            verbose_eval=True
                            )
        mean_auc = pd.Series(cv_results['auc-mean']).max()
        boost_rounds = pd.Series(cv_results['auc-mean']).argmax()
        if mean_auc < max_auc:
            max_auc = mean_auc
            best_params['bagging_fraction'] = bagging_fraction
            best_params['bagging_freq'] = bagging_freq

params['bagging_fraction'] = best_params['bagging_fraction']
params['bagging_freq'] = best_params['bagging_freq']

param['is_unbalance']='true'

print(best_params)
### 训练

params['learning_rate']=0.01
gbm = lgb.train(
          params,                     # 参数字典
          lgb_train,                  # 训练集
          valid_sets=lgb_eval,        # 验证集
          num_boost_round=500,       # 迭代次数
          early_stopping_rounds=50    # 早停次数
          )


### 线下预测
print ("线下预测")
### 训练
params['learning_rate']=0.01
lgb.train(
          params,                     # 参数字典
          lgb_train,                  # 训练集
          valid_sets=lgb_eval,        # 验证集
          num_boost_round=2000,       # 迭代次数
          early_stopping_rounds=50    # 早停次数
          )
# 将参数写成字典下形式
#param = {
#    'max_depth':6,
#    'num_leaves':64,
#    'learning_rate':0.03,
#    'scale_pos_weight':1,
#    'num_threads':40,
#    'objective':'binary',
#    'bagging_fraction':0.7,
#    'bagging_freq':1,
#    'min_sum_hessian_in_leaf':100
#}
#
#
#param['is_unbalance']='true'
#param['metric'] = 'auc'

print('Start prams...')
# 训练 cv and train
#bst=lgb.cv(param,lgb_train, num_boost_round=100, nfold=3, early_stopping_rounds=30)
lg = lgb.LGBMClassifier(silent=False)  
param_dist = {"max_depth": [25,50, 75],  
              "learning_rate" : [0.01,0.05,0.1],  
              "num_leaves": [300,900,1200],  
              "n_estimators": [200]  
             }  
grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv = 5, scoring="roc_auc", verbose=5)  
grid_search.fit(X_train, y_train)  
grid_search.best_estimator_ 

#gbm = lgb.train(param,lgb_train,num_boost_round=200,valid_sets=lgb_eval,early_stopping_rounds=100) #num_boost_round=len(bst['auc-mean'])
###print('Save model...')
### 保存模型到文件
###gbm.save_model('model.txt')
##
#print('Start predicting...')
### 预测数据集
#pred = gbm.predict(X_for_test, num_iteration=gbm.best_iteration)
#
#print(metrics.roc_auc_score(Y_for_test,pred))
#
#
##网格搜索 取最优的阈值
##pred1=dict()
##for key in range(len(pred)):
##    pred1[key] = pred[key]
##
##for i in range(1,10):
##    pred2 = []
##    for j in pred1:
##        value = pred1.get(j)
##        if value > (i/10):
##            pred2.append(1)
##        else:
##            pred2.append(0)
##    print (metrics.precision_score(Y_for_test,np.array(pred2)))
##    print (metrics.recall_score(Y_for_test,np.array(pred2)))
##    print (str(metrics.f1_score(Y_for_test,np.array(pred2))) + '\n')
#
##print(metrics.classification_report(Y_for_test, pred1))
##print(metrics.confusion_matrix(Y_for_test, pred1))
### 评估模型
##print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
#lgb_score = pd.Series(gbm.feature_importance(),index=list(X_test.columns)).sort_values(ascending=False)           
##lgb_score.plot(kind='bar', title='Feature Importances')
#print (lgb_score)

#print('saving t_fea...')
#t_fea.to_csv('D:/run/flight_all_feature.csv')
