# %% aaa
print("hello!")

#

# %%
# =================================================
# ライブラリ読み込み #
# =================================================

import re
import pickle
import gc
import os

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

import warnings
warnings.filterwarnings('ignore')



# %%
#フォルダの移動(確認)

print(f'現在:{os.getcwd()}')
os.chdir('/tmp/work/src/input/Home Credit Default Risk')
print(f'移動後:{os.getcwd()}')

for dirname, _, filenames in os.walk(os.getcwd()):
    for filename in filenames:
        # print(os.path.join(dirname,filename))
        print('='*20)
        print(filename)

#%%
os.getcwd()




# %%
# ファイルの読み込み #
# =================================================

application_train = pd.read_csv("application_train.csv")
print(application_train.shape)
application_train.head()



# %%
# メモリ削減関数 #
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
                

# %%
#実行
application_train = reduce_mem_usage(application_train)




# %%
# =================================================
# データセット #
# =================================================
x_train = application_train.drop(columns=['TARGET','SK_ID_CURR'])
y_train =application_train['TARGET']
id_train = application_train[['SK_ID_CURR']]

# %%
#カテゴリ型に変換
for col in x_train.columns:
    if x_train[col].dtype == 'O':
        x_train[col] = x_train[col].astype('category')


# %%
# バリデーション
# =================================================
print(f'mean:{y_train.mean():.4f}')
y_train.value_counts()

# %%
#層化分裂したバリデーションのindexのリスト作成
# =================================================
cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(x_train, y_train))

#indexの確認：fold=0の学習データ
print('index(train):',cv[0][0])

#indexの確認：fold=0の検証データ
print('index(train):',cv[0][1])


# %%
# モデル学習
# =================================================
#0
cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(x_train, y_train))

nfold = 0
idx_tr, idx_va = cv[nfold][0], cv[nfold][1]

#1.学習データと検証データに分離
x_tr, y_tr, id_tr = x_train.loc[idx_tr, :], y_train[idx_tr], id_train.loc[idx_tr, :]
x_va, y_va, id_va = x_train.loc[idx_va, :], y_train[idx_va], id_train.loc[idx_va, :]
print(x_tr.shape, y_tr.shape, id_tr.shape)
print(x_va.shape, y_va.shape, id_va.shape)



# %%
#2.モデル学習
# =================================================
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
model = lgb.LGBMClassifier(**params)
model.fit(x_tr,
          y_tr,
          eval_set = [(x_tr, y_tr), (x_va, y_va)],
          callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(100),
            ],
        )


# %%
print(f'今:{os.getcwd()}')
os.chdir('/tmp/work/src/exp')
print(f'今:{os.getcwd()}')

# %%
# モデルの保存
with open('model_lgb_fold0.pickle', 'wb') as f:
    pickle.dump(model, f, protocol=4)


#3.モデル評価
#学習データの推論値取得とROC計算
y_tr_pred = model.predict_proba(x_tr)[:,1]
metric_tr = roc_auc_score(y_tr, y_tr_pred)

#検証データの推論値のROC計算
y_va_pred = model.predict_proba(x_va)[:,1]
metric_va = roc_auc_score(y_va, y_va_pred)

#評価値を入れる変数の作成
metrics = []

#評価値を格納
metrics.append([nfold, metric_tr,metric_va])

#結果の表示
print(f'[AUC] tr:{metric_tr:.4f}, va:{metric_va:.4f}')

# %%
#4.OOFデータの推論値取得
#OOFの推論値を入れる変数の作成
train_oof = np.zeros(len(x_train))

#検証データのindexに推論値を格納
train_oof[idx_va] = y_va_pred

# %%
#5.説明変数の重要度取得
#重要度の取得
imp_fold = pd.DataFrame({'col':x_train.columns, 'imp':model.feature_importances_, 'nfold':nfold})

#確認（重要度の上位10個）
from IPython.core.display import display
display(imp_fold.sort_values('imp', ascending=False)[:10])

#重要度を格納する5-fold用データフレームの作成
imp = pd.DataFrame()

#imp_foldを5-fold用データフレームに結合
imp = pd.concat([imp, imp_fold])



# %%
#モデル評価

#リスト型をarray型に変換
metrics = np.array(metrics)
print(metrics)

#学習/検証データの評価値の平均値と標準偏差を算出
print(f'[cv] tr:{metrics[:,1].mean():.4f}+-{metrics[:,1].std():.4f}, va:{metrics[:,2].mean():.4f}+-{metrics[:,1].std():.4f}')

#oofの評価値を算出
print(f'[oof]{roc_auc_score(y_train, train_oof):.4f}')


# %%
train_oof = pd.concat([
    id_train,
    pd.DataFrame({'true': y_train, 'pred': train_oof}),
], axis=1)
train_oof.head()


# %%
imp = imp.groupby('col')['imp'].agg(['mean', 'std']).reset_index(drop=False)
imp.columns = ['col', 'imp', 'imp_std']
imp.head()


# %% [markdown]
## 学習関数
# %%
#学習関数の定義
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

    #保存フォルダの移動
    os.chdir('/tmp/work/src/output')
    print(f'保存場所:{os.getcwd()}')
    
    #1.学習データと検証データに分離
    for nfold in list_nfold:
        print('-'*20,nfold,'-'*20)

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
                  ],
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
    print(imp.sort_values('imp',ascending=False)[:10])


  
    return train_oof, imp, metrics

#%%
#学習処理の実行
#ハイパーパラメータの設定
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

#学習の実行
train_oof, imp, metrics = train_lgb(x_train,
                                    y_train,
                                    id_train,
                                    params,
                                    list_nfold=[0,1,2,3,4],
                                    n_splits=5,
                                    )


# %%
#重要度の確認
display(imp.sort_values('imp',ascending=False)[:10])



# %% [markdown]
## モデル推論


# %%モデル推論
#現在地の移動
print(f'今まで: {os.getcwd()}')
os.chdir('../input/Home Credit Default Risk')
print(f'NOW: {os.getcwd()}')

#%%
#テストファイルの読み込み
application_test = pd.read_csv('application_test.csv')
application_test = reduce_mem_usage(application_test)

#データセット
x_test = application_test.drop(columns=['SK_ID_CURR'])
id_test = application_test[['SK_ID_CURR']]

#カテゴリ変数をcategory型に
for col in x_test.columns:
    if x_test[col].dtype == 'O':
        x_test[col] = x_test[col].astype('category')

# %%
# os.getcwd()
os.chdir('../output')
os.getcwd()

# %%
#学習モデルの読み込み
with open('model_lgb_fold0.pickle', 'rb') as f:
    model = pickle.load(f)


# %%
#推論
test_pred_fold = model.predict_proba(x_test)[:, 1]

#推論値を格納する変数を作成
test_pred = np.zeros((len(x_test), 5))

#1-fold目の推論値を格納
test_pred[:,0] = test_pred_fold


# %%
#平均して推論値の算出
test_pred_mean = test_pred.mean(axis=1)

#推論値のデータフレーム作成
df_test_pred = pd.concat([
    id_test,
    pd.DataFrame({'pred': test_pred_mean}),
    ],axis=1) 
df_test_pred.head() 


# %% [markdown]
## 推論関数の定義
# %%
def predict_lgb(input_x,
                input_id,
                list_nfold=[0,1,2,3,4],
                ):
    
    os.chdir('/tmp/work/src/output')
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

# %%

test_pred = predict_lgb(x_test,
                        id_test,
                        list_nfold=[0,1,2,3,4],
                        )
# %%
test_pred.head()
# %%
df_submit = test_pred.rename(columns={'pred':'TARGET'})
print(df_submit.shape)
display(df_submit.head())

df_submit.to_csv('submission_baseline.csv', index=None)
# %%

os.getcwd()



# %% Dataset
# Dataset
# =================================================
# %%
