

#%%
# ======================================================

# ======================================================




# %%
#データセット
# =================================================

set_file = application_train
x_train = set_file.drop(columns=[target_columns,sub_index])
y_train =set_file[target_columns]
id_train = set_file[[sub_index]]




# %%
#カテゴリ型に変換
# =================================================
x_train = data_pre01(x_train)
# for col in x_train.columns:
#     if x_train[col].dtype == 'O':
#         x_train[col] = x_train[col].astype('category')



#%%
#学習処理の実行
# =================================================

train_oof, imp, metrics = train_lgb(x_train,
                                    y_train,
                                    id_train,
                                    params,
                                    list_nfold=[0,1,2,3,4],
                                    n_splits=5,
                                    )





#%%
#importance上位20
# =================================================
imp_sort = imp.sort_values('imp',ascending=False)
display(imp_sort[:20])
imp_sort.to_csv(f'importance_{file_name}.csv', index=None)


#%%
#テストファイルの読み込み
# =================================================
application_test = pd.read_csv(input_path+'application_test.csv')
application_test = reduce_mem_usage(application_test)

#データセット
set_file = application_test
x_test = set_file.drop(columns=[sub_index])
id_test = set_file[[sub_index]]

#カテゴリ変数をcategory型に
x_test = data_pre01(x_test)



# %%
# 推論
# =================================================
test_pred = predict_lgb(x_test,
                        id_test,
                        list_nfold=[0,1,2,3,4],
                        )
# %%
test_pred.head()
# %%
# submitファイルの出力
# =================================================
df_submit = test_pred.rename(columns={'pred':target_columns})
print(df_submit.shape)
display(df_submit.head())

df_submit.to_csv(f'submission_{file_name}.csv', index=None)

