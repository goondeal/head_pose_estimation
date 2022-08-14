import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from utils import DataExtractor, split_train_dev_x_y


class HeadPoseEstimator:
    def __init__(self):
        self.pca = PCA(n_components=0.98)
        self.scaler = StandardScaler()
        self.pitch_model = RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_leaf=3)
        self.yaw_model = RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_leaf=3)
        self.roll_model = RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_leaf=3)
        # self.pitch_model = SVR(C=10)
        # self.yaw_model = SVR(C=500)
        # self.roll_model = SVR(C=10)


    def _normalize_features(self, x):
        # Normalization Step:
        # consider the nose point as the center to avoid the change
        # of faces position in images. (simply subtract all points from the nose point)
        X = x.copy()
        idx = np.arange(0, X.shape[1], 1)
        X[:, idx % 2 == 0] -= X[:, 0].reshape(-1, 1)
        X[:, idx % 2 == 1] -= X[:, 1].reshape(-1, 1)
        # devide by the distance_from_nose_to_chin to normalize the distance from the face to the camera.
        chin_point_idx = 377
        chin_x = (chin_point_idx + 1) * 2
        chin_y = chin_x + 1
        d = np.sqrt(chin_x**2 + chin_y**2)
        X = X / d
        return X

    def fit(self, x, y):
        X = self._normalize_features(x)
        X = self.pca.fit_transform(X)
        X = self.scaler.fit_transform(X)

        pd.DataFrame(data=X, columns=[
                     f'x{i}' for i in range(X.shape[1])]).to_csv('X.csv')

        self.pitch_model = self.pitch_model.fit(X, y[:, 0])
        self.yaw_model = self.yaw_model.fit(X, y[:, 1])
        self.roll_model = self.roll_model.fit(X, y[:, 2])
        return self

    def predict(self, x):
        X = self._normalize_features(x)
        X = self.pca.transform(X)
        X = self.scaler.transform(X)

        return [
            self.pitch_model.predict(X),
            self.yaw_model.predict(X),
            self.roll_model.predict(X)
        ]

    def report(self, y, h):
        print('Pitch r2_score =', r2_score(y[:, 0], h[0]))
        print('Yaw r2_score =', r2_score(y[:, 1], h[1]))
        print('Roll r2_score =', r2_score(y[:, 2], h[2]))


if __name__ == '__main__':
    np.random.seed(42)
    df_path = 'data.csv'

    try:
        df = pd.read_csv(df_path, index_col=0)
        print('df exists')
    except:
        extractor = DataExtractor(
            '/home/ahmad/projects/python/head_pose_estimation/AFLW2000-3D/AFLW2000')
        df = extractor.get_data_df()
        df.to_csv(df_path)

    X_train, Y_train, X_dev, Y_dev = split_train_dev_x_y(df)

    model = HeadPoseEstimator()
    model = model.fit(X_train, Y_train)
    # Measure errors on train set
    y_hat = model.predict(X_train)
    model.report(Y_train, y_hat)

    # validate results on validation set
    h = model.predict(X_dev)
    model.report(Y_dev, h)
