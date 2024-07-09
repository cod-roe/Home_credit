
#%%


import os

# %%
os.getcwd()
# %%
#ファイル名取得
file_path = "/tmp/work/src/exp/sample.py"
def filename(file_path):
  file_name = os.path.splitext(os.path.basename(file_path))[0]
  # print(file_name)
  return file_name

# %%
#output配下に現在のファイル名のフォルダを作成
def namefolder(file_path):
  # file_path = "/tmp/work/src/exp/sample.py"
  file_name = os.path.splitext(os.path.
basename(file_path))[0]

  os.chdir('/tmp/work/src/output')
  os.makedirs(file_name)
  return file_name

file_path = "/tmp/work/src/exp/sample.py"

file_name = namefolder(file_path)
# %%
# %%
file_name = filename(file_path)
# %%
print(file_name)
# %%今の日付
import datetime
dt_now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
# %%
print(dt_now.strftime('%Y年%m月%d日 %H:%M:%S'))
# %%
