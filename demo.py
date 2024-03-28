import numpy as np
from tsai.all import TSRegressor, TSStandardize, SlidingWindow, MSELossFlat, rmse, load_learner, TSRegression, TimeSplitter
import matplotlib.pyplot as plt
import pandas as pd
import os
from tsai.data.all import *
from tsai.all import *
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


X = np.load('./prepared_data.npy', allow_pickle=True)
y_true = X[0]

scaler = StandardScaler()
X = scaler.fit_transform(X)


def tsai_process(X, model_name, total_epoch, window_length=10, horizon=5):
    # 使用 SlidingWindow 处理时间序列数据
    # X, y = SlidingWindow(window_length, get_y=0, horizon=horizon, stride=None)(X)

    splits = get_splits(y, test_size=0.3, stratify=True, random_state=23, shuffle=True,  show_plot=False)
    tfms = [None, [TSRegression()]]
    batch_tfms = TSStandardize(by_sample=True)

    learn = TSRegressor(X, y, splits=splits, path='models', arch=model_name, tfms=tfms,
                        batch_tfms=batch_tfms, metrics=mse)

    # 训练模型
    learn.fit_one_cycle(total_epoch, 1e-3)
    learn.export(f'{model_name}.pkl')
    # Ensure the directory for saving the model exists



    # learn.plot_metrics(save_path=save_path + f'{model_name}_{total_epoch}_learning_curves.png')
    learn.plot_metrics()


    plt.savefig( f'{model_name}_{total_epoch}_metrics.png')

def tsai_predict_process(X,  model_name, save_path,y_true ):
    # learn = load_learner('models/1.pkl')
    learn = load_learner(save_path + model_name +'.pkl')
    test_probas, test_targets, test_preds = learn.get_X_preds(X, with_decoded=True)
    test_preds = np.array(test_preds)
    test_preds = test_preds.astype(np.float32)

    # 计算准确率
    r2 = r2_score(y_true,test_preds)

    return r2, test_preds


window_length = 10
horizon = 5

X, y = SlidingWindow(10, get_y=0, horizon=5, stride=None)(X)

# 调用训练过程
# 调用 tsai_process 函数时传递模型名参数
tsai_process(X, model_name='XceptionTimePlus', total_epoch=50,  window_length=10, horizon=5)


r2 ,test_preds= tsai_predict_process(X, model_name='XceptionTimePlus', save_path='./models/', y_true=y)
print(f'R2 Score: {r2}')



