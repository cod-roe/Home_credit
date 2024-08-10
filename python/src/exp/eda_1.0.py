# %% [markdown]
## EDA データの理解、集計、可視化
#=================================================


# %%
# # ファイルの理解とER図 


#%%
#ライブラリ読み込み
# =================================================
import datetime
import gc
import re
import os
import pickle
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
sns.set(font="IPAexGothic")
# %matplotlib inline
import ydata_profiling as pdp


#sckit-learn
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

#lightGBM
import lightgbm as lgb






#%%
#Config
# =================================================

######################
# serial #
######################
serial_number = 2


######################
# Data #
######################
input_path = '/tmp/work/src/input/Home Credit Default Risk/'
file_path = "/tmp/work/src/script/eda_1.0.py"
file_name = os.path.splitext(os.path.basename(file_path))[0]


######################
# Dataset #
######################
target_columns = 'TARGET'
sub_index = 'SK_ID_CURR'

######################
# ハイパーパラメータの設定
######################
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 32,
    'n_estimators':100000,
    'random_state': 123,
    'importance_type': 'gain',
}


# =================================================
# Utilities #
# =================================================

# 今の日時
def dt_now():
    dt_now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    return dt_now



# %%
#メモリ削減関数
# =================================================
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage of dataframe is {start_mem:.2f} MB')

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astypez(np.float64)
        else:
            pass
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage after optimization is: {end_mem:.2f} MB')
    print(f'Decreased by {100*((start_mem - end_mem) / start_mem):.2f}%')

    return df
                

#%%
#前処理の定義 カテゴリ変数をcategory型に
# =================================================
def data_pre01(df):
    for col in df.columns:
        if df[col].dtype == 'O':
            df[col] = df[col].astype('category')
    print('カテゴリ変数をcategory型に変換しました')
    return df

                
# %%
#学習関数の定義
# =================================================
def train_lgb(input_x,
              input_y,
              input_id,
              params,
              list_nfold=[0,1,2,3,4],
              n_splits=5,
            ):
    
    metrics = []
    imp = pd.DataFrame()
    train_oof = np.zeros(len(input_x))

    # cross-validation
    cv = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123).split(input_x, input_y))


    #output配下に現在のファイル名のフォルダを作成し、移動
    os.chdir('/tmp/work/src/output')
    if not os.path.isdir(file_name):
        os.makedirs(file_name)
        print(f'{file_name}フォルダ作成しました')
    os.chdir('/tmp/work/src/output/'+file_name)
    print(f'保存場所: {os.getcwd()}')
    
    #1.学習データと検証データに分離
    for nfold in list_nfold:
        print('-'*20,nfold,'-'*20)
        print(dt_now().strftime('%Y年%m月%d日 %H:%M:%S'))

        idx_tr, idx_va = cv[nfold][0], cv[nfold][1]

        x_tr, y_tr, id_tr = input_x.loc[idx_tr, :], input_y[idx_tr], input_id.loc[idx_tr, :]
        x_va, y_va, id_va = input_x.loc[idx_va, :], input_y[idx_va], input_id.loc[idx_va, :]

        print(x_tr.shape, x_va.shape)

        #train
        model = lgb.LGBMClassifier(**params)
        model.fit(x_tr,
                  y_tr,
                  eval_set = [(x_tr, y_tr), (x_va, y_va)],
                  callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=True),
                    lgb.log_evaluation(100), 
                  ]
                  )
        
        # モデルの保存
        fname_lgb = f'model_lgb_fold{nfold}.pickle'
        with open(fname_lgb, 'wb') as f:
            pickle.dump(model, f, protocol=4)

        #evaluate
        y_tr_pred = model.predict_proba(x_tr)[:,1]
        y_va_pred = model.predict_proba(x_va)[:,1]
        metric_tr = roc_auc_score(y_tr, y_tr_pred)
        metric_va = roc_auc_score(y_va, y_va_pred)
        metrics.append([nfold, metric_tr, metric_va])
        print(f'[auc] tr:{metric_tr:.4f}, va:{metric_va:.4f}')

        #oof
        train_oof[idx_va] = y_va_pred

        #imp
        _imp = pd.DataFrame({'col':input_x.columns, 'imp':model.feature_importances_, 'nfold':nfold})
        imp = pd.concat([imp, _imp])
    
    print('-'*20,'result','-'*20)

    #metric
    metrics = np.array(metrics)
    print(metrics)
    print(f'[cv] tr:{metrics[:,1].mean():.4f}+-{metrics[:,1].std():.4f}, \
          va:{metrics[:,2].mean():.4f}+-{metrics[:,1].std():.4f}')
    
    print(f'[oof]{roc_auc_score(input_y, train_oof):.4f}')
    
    #oof
    train_oof = pd.concat([
        input_id,
        pd.DataFrame({ 'pred': train_oof}),
    ], axis=1)

    #importance
    imp = imp.groupby('col')['imp'].agg(['mean', 'std']).reset_index(drop=False)
    imp.columns = ['col', 'imp', 'imp_std']

    print('-'*20,'importance','-'*20)
    print(imp.sort_values('imp',ascending=False)[:10])
  
    return train_oof, imp, metrics



# %%
# 推論関数の定義 =================================================
def predict_lgb(input_x,
                input_id,
                list_nfold=[0,1,2,3,4],
                ):
    
    #モデル格納場所へ移動
    os.chdir('/tmp/work/src/output/'+file_name)
    
    pred = np.zeros((len(input_x), len(list_nfold)))

    for nfold in list_nfold:
        print('-'*20,nfold,'-'*20)

        fname_lgb =f'model_lgb_fold{nfold}.pickle'
        with open(fname_lgb, 'rb') as f:
            model = pickle.load(f)

        #推論
        pred[:,nfold] = model.predict_proba(input_x)[:, 1]
        

    #平均値算出
    pred = pd.concat([
        input_id,
        pd.DataFrame({'pred':pred.mean(axis=1)}),
    ], axis=1)
    print('Done.')

    return pred


# %% [markdown]
## 分析start!
#%%
#出力表示数増やす
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)


#%%
#ファイルの確認
# =================================================
datainput = []
for dirname, _, filenames in os.walk(input_path):
    for i, datafilename in enumerate(filenames):
        # print(os.path.join(dirname,filename))
        print('='*40)
        print(i,datafilename)
        datainput.append(datafilename[:-4])
print(datainput)
          
#%%
#ファイルの読み込み application_test
# =================================================

# application_test = reduce_mem_usage(pd.read_csv(input_path+"application_test.csv"))
# print(application_test.shape)
# display(application_test.head())
 

#%%
#ファイルの読み込み application_train
# =================================================

application_train = reduce_mem_usage(pd.read_csv(input_path+"application_train.csv"))
print('application_train')
print(application_train.shape)
application_train.head()

#%%
#ファイルの読み込み bureau
# =================================================

# bureau = reduce_mem_usage(pd.read_csv(input_path+"bureau.csv"))
# print('bureau')
# print(bureau.shape)
# bureau.head()

#%%
#ファイルの読み込み bureau_balance
# =================================================

# bureau_balance = reduce_mem_usage(pd.read_csv(input_path+"bureau_balance.csv"))
# print('bureau_balance')
# print(bureau_balance.shape)
# bureau_balance.head()


#%%
#ファイルの読み込み credit_card_balance
# =================================================

# credit_card_balance = reduce_mem_usage(pd.read_csv(input_path+"credit_card_balance.csv"))
# print('credit_card_balance')
# print(credit_card_balance.shape)
# credit_card_balance.head()

# #%%
# #ファイルの読み込み installments_payments
# # =================================================

# installments_payments = reduce_mem_usage(pd.read_csv(input_path+"installments_payments.csv"))
# print('installments_payments')
# print(installments_payments.shape)
# installments_payments.head()


# #%%
# #ファイルの読み込み POS_CASH_balance
# # =================================================

# POS_CASH_balance = reduce_mem_usage(pd.read_csv(input_path+"POS_CASH_balance.csv"))
# print('POS_CASH_balance')
# print(POS_CASH_balance.shape)
# POS_CASH_balance.head()


# #%%
# #ファイルの読み込み previous_application
# # =================================================

# previous_application = reduce_mem_usage(pd.read_csv(input_path+"previous_application.csv"))
# print('previous_application')
# print(previous_application.shape)
# previous_application.head()

# %%
#メモリ削減実行
# =================================================
# application_train = reduce_mem_usage(application_train)




# %%
application_train.info()

# %%
application_train.head()
# %%
application_train.shape

# %%
application_train.dtypes

# %%
application_train.columns

# %%
application_train.isna().sum()

# %%
application_train.describe().T
# %%
application_train.describe(exclude='number').T


# %%
# sns.countplot(data=application_train,x='TARGET')


# # %%
# display(application_train['TARGET'].value_counts())
# # %%
# display(application_train['TARGET'].value_counts()/len(application_train['TARGET']))


# # %%
# sns.countplot(data=application_train,x='FLAG_OWN_CAR',hue='TARGET')

# # %%
# pd.crosstab(application_train['FLAG_OWN_CAR'],application_train['TARGET'])
# # %%
# pd.crosstab(application_train['FLAG_OWN_CAR'],application_train['TARGET'],normalize='index')

# # %%
# sns.histplot(application_train['AMT_INCOME_TOTAL'],kde=False)
# # %%
# pd.crosstab(application_train['CNT_CHILDREN'],application_train['TARGET'])
# # %%
# pd.crosstab(application_train['CNT_CHILDREN'],application_train['TARGET'],normalize='index')
# %%
display(application_train['AMT_INCOME_TOTAL'].value_counts()/len(application_train['AMT_INCOME_TOTAL']))
# %%
