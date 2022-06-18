import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from utils import DataCleaner


class HeadPoseEstimator:
    def __init__(self):
        self.pitch_model = SVR(C=100)
        self.yaw_model = SVR(C=100)
        self.roll_model = SVR(C=100)

    def fit(self, x, y):
        self.pitch_model = self.pitch_model.fit(x, y[:, 0])
        self.yaw_model = self.yaw_model.fit(x, y[:, 1])
        self.roll_model = self.roll_model.fit(x, y[:, 2])
        return self

    def predict(self, x):
        return [
            self.pitch_model.predict(x),
            self.yaw_model.predict(x),
            self.roll_model.predict(x)
            ]

    def report(self, y, h):
        print('Pitch r2_score =', r2_score(y[:, 0] , h[0]))
        print('Yaw r2_score =', r2_score(y[:, 1] , h[1]))
        print('Roll r2_score =', r2_score(y[:, 2] , h[2]))


if __name__ == '__main__':
    np.random.seed(42)
    df_path = 'cleaned_data_mp.csv'
    # df_path = 'cleaned_data_21pts.csv'

    try:
        df = pd.read_csv(df_path, index_col=0)
    except:
        cleaner = DataCleaner(
            '/home/ahmad/projects/python/head_pose_estimation/AFLW2000-3D/AFLW2000')
        df = cleaner.get_cleaned_df()
        df.to_csv(df_path)

    # print(df.head())
    # print(df.describe())

    def split_x_y(data):
        return data[:, :-3], data[:, -3:]

    data_train, data_test = train_test_split(df, train_size=0.90)
    # data_dev, data_test = train_test_split(data_dev_test, train_size=0.50, shuffle=False)
    
    X_train, Y_train = split_x_y(data_train.values)
    # X_dev, Y_dev = split_x_y(data_dev.values)
    X_test, Y_test = split_x_y(data_test.values)

    model = HeadPoseEstimator()
    model = model.fit(X_train, Y_train)
    # validate results on validation set
    h = model.predict(X_test)

    model.report(Y_test, h)
