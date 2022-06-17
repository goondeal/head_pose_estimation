import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from utils import DataCleaner, split_train_dev_x_y, draw_axis
from xgboost import XGBRegressor
import dlib
import cv2
from imutils import face_utils


class HeadPoseEstimator:
    def __init__(self):
        self.pitch_model = XGBRegressor(n_estimators=1000, nthread=4)
        self.yaw_model = XGBRegressor(n_estimators=1000, nthread=4)
        self.roll_model = XGBRegressor(n_estimators=1000, nthread=4)

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

    def report(self, X_train, Y_train, X_dev, Y_dev):
        print('\nAccuracy on training set: ')
        h_pitch = self.pitch_model.predict(X_train)
        h_yaw = self.yaw_model.predict(X_train)
        h_roll = self.roll_model.predict(X_train)
        print('Pitch r2_score =', r2_score(Y_train[:, 0] , h_pitch))
        print('Yaw r2_score =', r2_score(Y_train[:, 1] , h_yaw))
        print('Roll r2_score =', r2_score(Y_train[:, 2] , h_roll))

        print('\nAccuracy on dev set: ')
        h_pitch = self.pitch_model.predict(X_dev)
        h_yaw = self.yaw_model.predict(X_dev)
        h_roll = self.roll_model.predict(X_dev)
        print('Pitch r2_score =', r2_score(Y_dev[:, 0] , h_pitch))
        print('Yaw r2_score =', r2_score(Y_dev[:, 1] , h_yaw))
        print('Roll r2_score =', r2_score(Y_dev[:, 2] , h_roll))



if __name__ == '__main__':
    np.random.seed(42)
    df_path = 'cleaned_data_68pts.csv'

    try:
        df = pd.read_csv(df_path, index_col=0)
    except:
        print(f'file {df_path} does not exist, generating df ...')
        cleaner = DataCleaner(
            '/home/ahmad/projects/python/head_pose_estimation/AFLW2000-3D/AFLW2000')
        df = cleaner.get_cleaned_df()
        df.to_csv(df_path)

    X_train, Y_train, X_dev, Y_dev = split_train_dev_x_y(df)

    model = HeadPoseEstimator()
    model = model.fit(X_train, Y_train)
    # model.report(X_train, Y_train, X_dev, Y_dev)

    # Let's go life
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('/home/ahmad/projects/python/head_pose_estimation/data')
    
    cap = cv2.VideoCapture(0)
    # points = {17, 19, 21, 22, 24, 26, 36, 39, 42, 45, 31, 30, 35, 48, 66, 54, 8, 2, 14}

    while True:
        success, image = cap.read()

        # Flip the image horizontally for a later selfie-view display
        # Also convert the color space from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # To improve performance
        # image.flags.writeable = False

        # Get the result
        rects = detector(image, 1)
        # loop over the face detections
        for rect in rects:
            
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(image, rect)
            shape = face_utils.shape_to_np(shape)
            nose = shape[30]
            features = DataCleaner()._get_features(shape)
            features = np.delete(features, [30, 98], axis=1)
            pitch, yaw, roll = model.predict(features)
            draw_axis(image, pitch, yaw, roll, nose)
            # for n in points: # range(1, 68):
            #     x = shape.part(n).x
            #     y = shape.part(n).y
            #     cv2.circle(image, (x, y), 1, (0, 255, 255), 1)

        
            
        # To improve performance
        # image.flags.writeable = True

        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Face Landmarks", image)

        key = 0xFF & cv2.waitKey(1)
        if key & 0xFF == 27: #Exit program when the user presses 'esc'
            break
        
    cv2.destroyAllWindows()
    cap.release()

