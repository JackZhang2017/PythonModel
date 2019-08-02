# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
%matplotlib inline

#载入数据:
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')
train.shape, test.shape
train.dtypes
train.head(5)
#合成一个总的data
train['source']= 'train'
test['source'] = 'test'
data=pd.concat([train, test],ignore_index=True)
data.shape
data.apply(lambda x: sum(x.isnull()))
var = ['Gender','Salary_Account','Mobile_Verified','Var1','Filled_Form','Device_Type','Var2','Source']
for v in var:
    print '\n%s这一列数据的不同取值和出现的次数\n'%v
    print data[v].value_counts()
len(data['City'].unique())
data.drop('City',axis=1,inplace=True)
#DOB是出生的具体日期，咱们要具体日期作用没那么大，年龄段可能对我们有用，所有算一下年龄好了
data['DOB'].head()
#创建一个年龄的字段Age
data['Age'] = data['DOB'].apply(lambda x: 115 - int(x[-2:]))
data['Age'].head()
#把原始的DOB字段去掉:
data.drop('DOB',axis=1,inplace=True)
#EMI_Load_Submitted字段处理
data.boxplot(column=['EMI_Loan_Submitted'],return_type='axes')
#好像缺失值比较多，干脆就开一个新的字段，表明是缺失值还是不是缺失值
data['EMI_Loan_Submitted_Missing'] = data['EMI_Loan_Submitted'].apply(lambda x: 1 if pd.isnull(x) else 0)
data[['EMI_Loan_Submitted','EMI_Loan_Submitted_Missing']].head(10)
#原始那一列就可以不要了
data.drop('EMI_Loan_Submitted',axis=1,inplace=True)
len(data['Employer_Name'].value_counts())
#丢掉
data.drop('Employer_Name',axis=1,inplace=True)
data.boxplot(column='Existing_EMI',return_type='axes')
data['Existing_EMI'].describe()
#缺省值不多，用均值代替
data['Existing_EMI'].fillna(0, inplace=True)
data.boxplot(column=['Interest_Rate'],return_type='axes')
#缺省值太多，也造一个字段，表示有无
data['Interest_Rate_Missing'] = data['Interest_Rate'].apply(lambda x: 1 if pd.isnull(x) else 0)
print data[['Interest_Rate','Interest_Rate_Missing']].head(10)
data.drop('Interest_Rate',axis=1,inplace=True)
#不！要！了！，是的，不要了！！！
data.drop('Lead_Creation_Date',axis=1,inplace=True)
data.head()
#找中位数去填补缺省值（因为缺省的不多）
data['Loan_Amount_Applied'].fillna(data['Loan_Amount_Applied'].median(),inplace=True)
data['Loan_Tenure_Applied'].fillna(data['Loan_Tenure_Applied'].median(),inplace=True)
data.head()
# 缺省值太多。。。是否缺省。。。
data['Loan_Amount_Submitted_Missing'] = data['Loan_Amount_Submitted'].apply(lambda x: 1 if pd.isnull(x) else 0)
data['Loan_Tenure_Submitted_Missing'] = data['Loan_Tenure_Submitted'].apply(lambda x: 1 if pd.isnull(x) else 0)
data.head()
#原来的字段就没用了
data.drop(['Loan_Amount_Submitted','Loan_Tenure_Submitted'],axis=1,inplace=True)
#没想好怎么用。。。不要了。。。
data.drop('LoggedIn',axis=1,inplace=True)
# 可能对接多个银行，所以也不要了
data.drop('Salary_Account',axis=1,inplace=True)
#和之前一样的处理，有或者没有
data['Processing_Fee_Missing'] = data['Processing_Fee'].apply(lambda x: 1 if pd.isnull(x) else 0)
#旧的字段不要了
data.drop('Processing_Fee',axis=1,inplace=True)
data['Source'] = data['Source'].apply(lambda x: 'others' if x not in ['S122','S133'] else x)
data['Source'].value_counts()
data.head()
data.describe()
data.apply(lambda x: sum(x.isnull()))
data.dtypes
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
var_to_encode = ['Device_Type','Filled_Form','Gender','Var1','Var2','Mobile_Verified','Source']
for col in var_to_encode:
    data[col] = le.fit_transform(data[col])
data.head()
data.dtypes
data = pd.get_dummies(data, columns=var_to_encode)
data.columns
train = data.loc[data['source']=='train']
test = data.loc[data['source']=='test']
train.drop('source',axis=1,inplace=True)
test.drop(['source','Disbursed'],axis=1,inplace=True)
train.to_csv('train_modified.csv',index=False)
test.to_csv('test_modified.csv',index=False)