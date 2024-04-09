import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.api._v2 import keras
from keras.datasets import boston_housing
from rich import print
from sklearn.model_selection import train_test_split
import pandas as pd

import params
from preprocessing import preprocess_dataset


def predict(dataset):
    model = keras.models.load_model(params.MODEL_FILE_PATH)

    model_1 = keras.models.load_model('model_rank_1.h5')
    model_2 = keras.models.load_model('model_rank_2.h5')
    model_3 = keras.models.load_model('model_rank_3.h5')
    model_4 = keras.models.load_model('model_rank_4.h5')


    pred = model.predict(dataset).flatten()

    pred_1 = model_1.predict(dataset).flatten()
    pred_2 = model_2.predict(dataset).flatten()
    pred_3 = model_3.predict(dataset).flatten()
    pred_4 = model_4.predict(dataset).flatten()

    return pred_1, pred_2, pred_3, pred_4


if __name__ == '__main__':
    df = pd.read_csv('data.csv', index_col=0)
    df_pre_rank = df[['開局順位_1', '開局順位_2', '開局順位_3', '開局順位_4']].values
    df = df[['開局順位_1', '開局順位_2', '開局順位_3', '開局順位_4', "f_1", "f_2", "f_3", "f_4", 
             '順目', '供託', '本場', 'ドラ_1', 'ドラ_2', 'ドラ_3', 'ドラ_4', '赤ドラ_1', '赤ドラ_2', '赤ドラ_3', '赤ドラ_4', '向聴数_1', '向聴数_2', '向聴数_3', '向聴数_4',
             '順位_1', '順位_2', '順位_3', '順位_4', '局収支_1', '局収支_2', '局収支_3', '局収支_4'
             ]]
    
    t = df[['順位_1', '順位_2', '順位_3', '順位_4']].values
    x = df.drop(labels=['開局順位_1', '開局順位_2', '開局順位_3', '開局順位_4', '順位_1', '順位_2', '順位_3', '順位_4', '局収支_1', '局収支_2', '局収支_3', '局収支_4'], axis=1).values

    train_data, test_data, train_labels, test_labels = train_test_split(x, t, test_size=0.2, random_state=0)
    data_len = len(test_labels)
    test_data, test_labels = test_data[:data_len], test_labels[:data_len]
    test_data = preprocess_dataset(dataset=test_data, is_training=False)
    #pred = predict(dataset=test_data)
    pred_1, pred_2, pred_3, pred_4 = predict(dataset=test_data)
    print('prediction, labels')

    cnt_same = 0
    cnt_diff = 0
    mae_same = 0
    mae_diff = 0
    for i in range(data_len):
        #pred_list = [np.round(pred[i], 2)]
        pred_list = [
            np.round(pred_1[i], 2),
            np.round(pred_2[i], 2),
            np.round(pred_3[i], 2),
            np.round(pred_4[i], 2)
        ]

        rank_and = set(df_pre_rank[i]) & set(test_labels[i])

        if len(rank_and) == 4:
            for j in range(4):
                mae_same += abs(pred_list[j] - test_labels[i][j])
            cnt_same += 1
        else:
            for j in range(4):
                mae_diff += abs(pred_list[j] - test_labels[i][j])
            cnt_diff += 1

    print('same mae: ' + str(round(mae_same / cnt_same, 8)) + ', ' + str(cnt_same))
    print('diff mae: ' + str(round(mae_diff / cnt_diff, 8)) + ', ' + str(cnt_diff))
