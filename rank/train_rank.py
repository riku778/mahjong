import tensorflow as tf
from keras.api._v2 import keras
from keras.models import Model
from keras.datasets import boston_housing
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.optimizers import Adam
from keras.utils import plot_model
import pandas as pd
from sklearn.model_selection import train_test_split

import params
from preprocessing import preprocess_dataset
from model import DNNModel


def main():
    df = pd.read_csv('data/data_ukeire.csv', index_col=0)

    df = df[[
        "f_1", "f_2", "f_3", "f_4",
        '受け入れ_1_0', '受け入れ_1_1', '受け入れ_1_2', '受け入れ_2_0', '受け入れ_2_1', '受け入れ_2_2', '受け入れ_3_0', '受け入れ_3_1', '受け入れ_3_2', '受け入れ_4_0', '受け入れ_4_1', '受け入れ_4_2',
        "順目", "場風", "局数", "積み棒", "供託",
        "立直_1", "立直_2", "立直_3", "立直_4", "副露_1" ,"副露_2", "副露_3", "副露_4",
        "ドラ_1", "ドラ_2", "ドラ_3", "ドラ_4",
        '向聴数_1', '向聴数_2', '向聴数_3', '向聴数_4', "順位_1", "順位_2", "順位_3", "順位_4", "局収支_1", "局収支_2", "局収支_3", "局収支_4"
        ]]
    #'受け入れ_1_0', '受け入れ_1_1', '受け入れ_1_2', '受け入れ_2_0', '受け入れ_2_1', '受け入れ_2_2', '受け入れ_3_0', '受け入れ_3_1', '受け入れ_3_2', '受け入れ_4_0', '受け入れ_4_1', '受け入れ_4_2',
    
    t = df[['順位_1', '順位_2', '順位_3', '順位_4']].values
    x = df.drop(labels=['順位_1', '順位_2', '順位_3', '順位_4', '局収支_1', '局収支_2', '局収支_3', '局収支_4'], axis=1).values

    #train_data, test_data, train_labels, test_labels = train_test_split(x, t, test_size=0.2, random_state=0)
    train_data = preprocess_dataset(dataset=x, is_training=True)
    
    model: Model = DNNModel().build()

    optimizer = tf.keras.optimizers.legacy.Adam()    
    model.compile(
        optimizer=optimizer,
        loss='mae' #学習において最小化したい関数
        #metrics=['mae'] #学習とは無関係にモデルの性能評価のために用意する関数
        )
    model.summary()
    
    def step_decay(epoch):
        x = 0.00001
        if epoch >= 230: x = 0.000001
        if epoch >= 280: x = 0.0000001
        return x
    
    callbacks = [
        EarlyStopping(patience=50),
        ModelCheckpoint(filepath=str('model/model4_rank.h5'), save_best_only=True),
        TensorBoard(log_dir=params.LOG_DIR),
        LearningRateScheduler(step_decay)
        ]
    
    model.fit(
        x=train_data,
        y=t,
        epochs=params.EPOCHS,
        validation_split=params.VALIDATION_SPLIT,
        callbacks=callbacks
        )
    
if __name__ == '__main__':
    main()
