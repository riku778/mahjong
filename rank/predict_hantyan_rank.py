import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.api._v2 import keras
from keras.datasets import boston_housing
from rich import print
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib


import params
from preprocessing import preprocess_dataset


def predict(dataset):
    model = keras.models.load_model('model/model4_rank.h5')
    predictions = model.predict(dataset)
    return predictions


if __name__ == '__main__':
    df = pd.read_csv('data/data_hantyan_1.csv', index_col=0)
    df = df[[
        "f_1", "f_2", "f_3", "f_4", 
        '受け入れ_1_0', '受け入れ_1_1', '受け入れ_1_2', '受け入れ_2_0', '受け入れ_2_1', '受け入れ_2_2', '受け入れ_3_0', '受け入れ_3_1', '受け入れ_3_2', '受け入れ_4_0', '受け入れ_4_1', '受け入れ_4_2',
        "順目", "場風", "局数", "積み棒", "供託",
        "立直_1", "立直_2", "立直_3", "立直_4", "副露_1" ,"副露_2", "副露_3", "副露_4",
        "ドラ_1", "ドラ_2", "ドラ_3", "ドラ_4",
        '向聴数_1', '向聴数_2', '向聴数_3', '向聴数_4', "順位_1", "順位_2", "順位_3", "順位_4", "局収支_1", "局収支_2", "局収支_3", "局収支_4"
        ]]
    
    
    t = df[['順位_1', '順位_2', '順位_3', '順位_4']].values
    x = df.drop(labels=['順位_1', '順位_2', '順位_3', '順位_4', '局収支_1', '局収支_2', '局収支_3', '局収支_4'], axis=1).values

    data_len = len(x)
    test_data, test_labels = x[:data_len], x[:data_len]
    test_data = preprocess_dataset(dataset=test_data, is_training=False)
    pred = predict(dataset=test_data)
    
    mark_point_x = []
    mark_point_through1 = []
    mark_point_through2 = []
    mark_point_through3 = []
    mark_point_through4 = []

    rank_through1 = []
    rank_through2 = []
    rank_through3 = []
    rank_through4 = []
    rank_through_pred1 = []
    rank_through_pred2 = []
    rank_through_pred3 = []
    rank_through_pred4 = []
    for i in range(1, data_len):
        if df['局数'][i-1] != df['局数'][i] or df['積み棒'][i-1] != df['積み棒'][i]:
            player = (4 - (df['局数'][i] % 4)) % 4
            mark_point_x.append(i)
            mark_point_through1.append(t[i, player])
            mark_point_through2.append(t[i, (player + 1) % 4])
            mark_point_through3.append(t[i, (player + 2) % 4])
            mark_point_through4.append(t[i, (player + 3) % 4])

    for i in range(data_len):
        player = (4 - (df['局数'][i] % 4)) % 4
        rank_through1.append(t[i, player])
        rank_through2.append(t[i, (player + 1) % 4])
        rank_through3.append(t[i, (player + 2) % 4])
        rank_through4.append(t[i, (player + 3) % 4])
        rank_through_pred1.append(pred[i, player])
        rank_through_pred2.append(pred[i, (player + 1) % 4])
        rank_through_pred3.append(pred[i, (player + 2) % 4])
        rank_through_pred4.append(pred[i, (player + 3) % 4])
        
    x_datas = range(1, data_len + 1)

    fig = plt.figure(figsize = (10,6), facecolor='lightblue')

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    
    ax1.plot(x_datas, rank_through1, label='順位')
    ax2.plot(x_datas, rank_through2, label='順位')
    ax3.plot(x_datas, rank_through3, label='順位')
    ax4.plot(x_datas, rank_through4, label='順位')

    ax1.plot(x_datas, rank_through_pred1, label='予想')
    ax2.plot(x_datas, rank_through_pred2, label='予想')
    ax3.plot(x_datas, rank_through_pred3, label='予想')
    ax4.plot(x_datas, rank_through_pred4, label='予想')

    ax1.set_xticks(np.arange(0, data_len, 100))
    ax2.set_xticks(np.arange(0, data_len, 100))
    ax3.set_xticks(np.arange(0, data_len, 100))
    ax4.set_xticks(np.arange(0, data_len, 100))


    ax1.set_title('一半荘における順位と予想(起家)', fontsize=20)  # グラフのタイトルを設定
    ax1.set_xlabel('順目', fontsize=16)  # x軸ラベルを設定
    ax1.set_ylabel('順位', fontsize=16)  # y軸ラベルを設定
    ax1.tick_params(labelsize=14)
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)

    ax1.scatter(mark_point_x, mark_point_through1)
    ax2.scatter(mark_point_x, mark_point_through2)
    ax3.scatter(mark_point_x, mark_point_through3)
    ax4.scatter(mark_point_x, mark_point_through4)

    ax1.set_ylim(0.9, 4.1)
    ax2.set_ylim(0.9, 4.1)
    ax3.set_ylim(0.9, 4.1)
    ax4.set_ylim(0.9, 4.1)

    plt.show() 
