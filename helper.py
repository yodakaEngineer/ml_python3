import pandas as pd

def deficiency_table(df):
    # isnull=各値に対してtrue,falseでNaNかどうかの判定を返す。それをsumで合計
    null_val = df.isnull().sum()
    # 欠損値の数/全要素数
    percent = 100 * null_val / len(df)
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(columns={0: '欠損数', 1: '%'})
    return print(kesson_table_ren_columns)
