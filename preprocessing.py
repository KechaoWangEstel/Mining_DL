import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# df_origin = pd.read_csv('pointWithAllParam.csv')
# # print(df_origin)
# scaler = MinMaxScaler()
# cols_to_scale = df_origin.columns.tolist()
# cols_to_scale.remove('system:index')
# cols_to_scale.remove('.geo')
# cols_to_scale.remove('area')
# cols_to_scale.remove('continent')
# cols_to_scale.remove('label')
# # print(cols_to_scale)
# # 选择列并应用归一化
# df_origin[cols_to_scale] = scaler.fit_transform(df_origin[cols_to_scale])
# # 打印归一化后的DataFrame
# # print(df_origin)
# df_origin.to_csv('normalized.csv', index=False, encoding='utf-8-sig')

def Normalize(filename):
    df_origin = pd.read_csv(filename)
    # print(df_origin)
    scaler = MinMaxScaler()
    cols_to_scale = df_origin.columns.tolist()
    cols_to_scale = df_origin.columns.tolist()
    cols_to_scale.remove('id')
    df_origin[cols_to_scale] = scaler.fit_transform(df_origin[cols_to_scale])
    # 打印归一化后的DataFrame
    # print(df_origin)
    df_origin.to_csv('normalized_mining.csv', index=False, encoding='utf-8-sig')

if __name__=='__main__':
    Normalize('pointWithAllParam_Mining.csv')

