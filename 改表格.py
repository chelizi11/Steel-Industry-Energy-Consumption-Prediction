import pandas as pd
import numpy as np

'''准备数据'''
def prepare_data():
    data = pd.read_excel('./Steel_industry_data.xlsx')
    # 替换字典
    dic = {
           'WeekStatus': {'Weekday': 0, 'Weekend': 1, },
           'Day_of_week': {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6},
           'Load_Type': {'Light_Load': 0, 'Medium_Load': 1, 'Maximum_Load': 2, },
           }
    # 使用replace方法进行替换
    data.replace(dic, inplace=True)
    # 转换为numpy数组
    data = np.array(data, dtype=np.float32)
    np.save('./prepared_data.npy', data)

prepare_data()
np.load('./prepared_data.npy')
print(np.load('./prepared_data.npy'))

