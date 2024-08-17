#%% 7.3.2データ前処理
#7-1ライブラリの読み込み
import numpy as np
import pandas as pd
import re
import pickle
import gc

#sckit-leatn
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

#LightGBM
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')
from IPython.core.display import display

# %%
#7-2 ファイルの読み込み
file_path = '../input/Home Credit Default Risk/'
app_train = pd.read_csv(file_path + 'application_train.csv')
print(app_train.shape)
app_train.head()
# %%
#7-3メモリ削減のための関数
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
                

# %%
#7-4 メモリ削減の実行
app_train = reduce_mem_usage(app_train)
# %%7.3.3 データセット作成
#データセットの作成
x_train = app_train.drop(columns=['SK_ID_CURR','TARGET'])
y_train = app_train['TARGET']
id_train = app_train[['SK_ID_CURR']] 


# %%
# 7-6 カテゴリ変数をcategory型に変換
for col in x_train.columns:
    if x_train[col].dtype == 'O':
        x_train[col] = x_train[col].astype('category')
# %%7.3.4バリデーション設計
# 7-7 1の割合とそれぞれの件数を確認
print(f'mean:{y_train.mean():.4f}')
y_train.value_counts()/len(y_train)
# %%
#7-8 バリデーションのindexリスト作成
cv = list(StratifiedKFold(n_splits=5,shuffle=True,random_state=123).split(x_train,y_train))

# 0fold目のindexのリスト作成
nfold = 0
idx_tr, idx_va = cv[nfold][0],cv[nfold][1]

#学習データと検証データに分離
x_tr, y_tr, id_tr = x_train.loc[idx_tr, :], y_train[idx_tr], id_train.loc[idx_tr, :]
x_va, y_va, id_va = x_train.loc[idx_va, :], y_train[idx_va], id_train.loc[idx_va, :]
print(x_tr.shape, y_tr.shape, id_tr.shape)
print(x_va.shape, y_va.shape, id_va.shape)

# %% スクリプト7-10：モデル学習
params={
    'boosting_type':'gbdt',
    'objective':'binary',
    'metric':'auc',
    'learning_rate':0.05,
    'num_leaves':32,
    'n_estimators':100000,
    'random_state':123,
    'importance_type':'gain',
}

#モデルの学習
model = lgb.LGBMClassifier(**params)
model.fit(
    x_tr,
    y_tr,
    eval_set=[(x_tr, y_tr),(x_va,y_va)],
    callbacks=[
		lgb.early_stopping(stopping_rounds=100,verbose=True),
		lgb.log_evaluation(100),
    ],
)
#モデル保存
with open('model_lgb_fold0.pickle', 'wb')as f:
    pickle.dump(model,f,protocol=4)

# %%
#7-11:モデル評価
#学習データの推論値取得とROC計算
y_tr_pred = model.predict_proba(x_tr)[:,1]
metric_tr = roc_auc_score(y_tr,y_tr_pred)

#検証データの推論値取得とROC計算
y_va_pred = model.predict_proba(x_va)[:,1]
metric_va = roc_auc_score(y_va, y_va_pred)

#評価値を入れる変数の作成（最初のfoldの時のみ）
metrics = []

#評価値を格納
metrics.append((nfold, metric_tr, metric_va))

print(f'[auc]tr:{metric_tr:.4f}, va:{metric_va:.4f}')
# %%
#7-12:oofデータの推論値取得
#oofの予測値を入れる変数の作成
train_oof = np.zeros(len(x_train))

#vaidデータnoindexに予測値を格納
train_oof[idx_va] = y_va_pred
# %%
#7-13:説明変数の重要度取得
#重要度の取得
imp_fold = pd.DataFrame({'col':x_train.columns, 'imp':model.feature_importances_, 'nfold':nfold})

#確認
display(imp_fold.sort_values('imp',ascending=False)[:10])

#重要度を格納する5fold用データフレームの作成
imp = pd.DataFrame()

# imp_foldを5fold用データフレームに結合
imp = pd.concat([imp,imp_fold])
# %%
# 7-14 モデル評価

# リスト型をarray型に変換
metrics = np.array(metrics)
print(metrics)

# 学習/検証データの評価値の平均値と標準偏差を算出
print(f'[cv]tr:{metrics[:,1].mean():.4f}+={metrics[:,1].std():.4f}, va:{metrics[:,2].mean():.4f}+={metrics[:,2].std():.4f}')

# oofの評価値を算出
print(f'[oof]{roc_auc_score(y_train, train_oof):.4f}')

# %% 
# 7-15 oofでーたの推論値取得（全foldのサマリ）
train_oof = pd.concat([
    id_train,
    pd.DataFrame({'true':y_train,'pred':train_oof})
],axis=1)
train_oof.head()

# %% 
# 7-16 説明変数の重要度取得（全foldのサマリ）
imp = imp.groupby('col')['imp'].agg(['mean','std']).reset_index(drop=False)
imp.columns = ['col','imp','imp_std']
imp.head()
# %% 
# 7-17 学習関数の定義
def train_lgb(
        input_x,
        input_y,
        input_id,
        params,
        list_nfold = [0,1,2,3,4],
        n_splits = 5,
        ):
    train_oof = np.zeros(len(input_x))
    metrics = []
    imp = pd.DataFrame()

    #cross-validation
    cv = list(StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=123).split(input_x,input_y))
    for nfold in list_nfold:
        print('-'*20,nfold,'-'*20)

        #make dataset
        idx_tr, idx_va = cv[nfold][0],cv[nfold][1]
        x_tr, y_tr, id_tr = input_x.loc[idx_tr, :], input_y[idx_tr], input_id.loc[idx_tr, :]
        x_va, y_va, id_va = input_x.loc[idx_va, :], input_y[idx_va], input_id.loc[idx_va, :]
        print(x_tr.shape, x_va.shape)

        #train
        model = lgb.LGBMClassifier(**params)
        model.fit(
            x_tr,
            y_tr,
            eval_set=[(x_tr, y_tr),(x_va,y_va)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100,verbose=True),
                lgb.log_evaluation(100),
            ],
        )
        
        fname_lgb = f'model_lgb_fold{nfold}.pickle'
        with open(fname_lgb, 'wb')as f:
            pickle.dump(model,f,protocol=4)

        #evaluate
        y_tr_pred = model.predict_proba(x_tr)[:,1]
        y_va_pred = model.predict_proba(x_va)[:,1]
        metric_tr = roc_auc_score(y_tr,y_tr_pred)
        metric_va = roc_auc_score(y_va, y_va_pred)
        metrics.append((nfold, metric_tr, metric_va))
        print(f'[auc]tr:{metric_tr:.4f}, va:{metric_va:.4f}')

        #oof
        train_oof[idx_va] = y_va_pred

        #imp
        _imp = pd.DataFrame({'col':input_x.columns, 'imp':model.feature_importances_, 'nfold':nfold})
        imp = pd.concat([imp,_imp])
    print('-'*20,'result','-'*20)
    #metric
    metrics = np.array(metrics)
    print(metrics)
    print(f'[cv]tr:{metrics[:,1].mean():.4f}+={metrics[:,1].std():.4f}, va:{metrics[:,2].mean():.4f}+={metrics[:,2].std():.4f}')
    print(f'[oof]{roc_auc_score(input_y, train_oof):.4f}')

    #oof
    train_oof = pd.concat([
        input_id,
        pd.DataFrame({'pred':train_oof})
    ],axis=1)

    #importance
    imp = imp.groupby('col')['imp'].agg(['mean','std']).reset_index(drop=False)
    imp.columns = ['col','imp','imp_std']
    
    return train_oof, imp, metrics

#%% 
# 7-18:学習処理の実行
#ハイパーパラメータの設定
params={
    'boosting_type':'gbdt',
    'objective':'binary',
    'metric':'auc',
    'learning_rate':0.05,
    'num_leaves':32,
    'n_estimators':100000,
    'random_state':123,
    'importance_type':'gain',
}

#学習の実行
train_oof, imp, metrics = train_lgb(
    x_train,
    y_train,
    id_train,
    params,
    list_nfold=[0,1,2,3,4],
    n_splits=5,
    )

# %%
# 7-19:説明変数の重要度の確認
imp.sort_values('imp',ascending=False)[:10]

# %%7.3.6 モデル推論
# 7-20:推論用データセットの作成

#ファイルの読み込み
app_test = pd.read_csv(file_path + 'application_test.csv')
app_test = reduce_mem_usage(app_test)

# データセットの作成
x_test = app_test.drop(columns = ['SK_ID_CURR'])
id_test = app_test[['SK_ID_CURR']]

# カテゴリ変数をcategory型に変換
for col in x_test.columns:
    if x_test[col].dtype == 'O':
        x_test[col] = x_test[col].astype('category')
# %%
# 7-21:学習済みモデルの読み込み
with open('model_lgb_fold0.pickle', 'rb') as f:
    model = pickle.load(f)
# %%
# 7-22:モデルを用いた推論
# 推論
test_pred_fold = model.predict_proba(x_test)[:,1]

# 推論値を格納する変数を作成
test_pred = np.zeros((len(x_test),5))

#1fold目の予測値を格納
test_pred[:, 0] = test_pred_fold

# %%
# 7-23:推論値用データセットの推論値算出

# 各foldの推論値の平均値を算出
test_pred_mean = test_pred.mean(axis=1)

# 推論値のデータフレームを作成
df_test_pred = pd.concat([
    id_test,
    pd.DataFrame({'pred':test_pred_mean}),
],axis=1)
df_test_pred.head()
# %%
# 7-24:推論関数の定義
def predict_lgb(
        input_x,
        input_id,
        list_nfold = [0,1,2,3,4],
        ):
    pred = np.zeros((len(input_x),len(list_nfold)))
    for nfold in list_nfold:
        print('-'*20,nfold,'-'*20)
        fname_lgb = f'model_lgb_fold{nfold}.pickle'
        with open(fname_lgb, 'rb') as f:
            model = pickle.load(f)
        pred[:,nfold] = model.predict_proba(input_x)[:,1]
    pred = pd.concat([
        input_id,
        pd.DataFrame({'pred':pred.mean(axis=1)}),
    ],axis=1)

    print('Done')

    return pred
# %%
test_pred = predict_lgb(
    x_test,
    id_test,
    list_nfold = [0,1,2,3,4],
)
# %% 7-26: 提出ファイルの作成
df_submit = test_pred.rename(columns={"pred":"TARGET"})
print(df_submit.shape)
display(df_submit.head())

# ファイル出力
# df_submit.to_csv("submission_baseline.csv", index=None)
# %%
