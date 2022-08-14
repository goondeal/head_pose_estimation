import cv2
import numpy as np
import scipy.io as sio
from math import cos, sin, pi
import mediapipe as mp
from glob import glob
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def split_x_y(data):
    ''' 
        Takes a matrix which its last 3 columns are the labels,
        Returns a tuple of 2 matrices (features and labels).
    '''
    return data[:, :-3], data[:, -3:]


def split_train_dev_x_y(df, train_size=0.9):
    '''
        Takes a data frame as input,
        split the data into train, dev sets,
        then, split each set into (features, labels) matrices.
    '''
    data_train, data_test = train_test_split(df, train_size=train_size)
    X_train, Y_train = split_x_y(data_train.values)
    X_dev, Y_dev = split_x_y(data_test.values)
    return X_train, Y_train, X_dev, Y_dev


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=100):
    yaw = -yaw

    if tdx == None or tdy == None:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right drawn in red
    x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
    y1 = size * (np.cos(pitch) * np.sin(roll) + np.cos(roll)
                 * np.sin(pitch) * np.sin(yaw)) + tdy

    # Y-Axis | drawn in green
    x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
    y2 = size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch)
                 * np.sin(yaw) * np.sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (np.sin(yaw)) + tdx
    y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    return img


class DataExtractor:
    def __init__(self, path):
        self.data_dir = Path(path)
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=2,
        )


    def _get_labels(self, mat):
        pose = mat.get('Pose_Para')[0, :3]
        return pose.reshape(1, -1)


    def extract_features_from_img(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Get the result
        results = self.face_mesh.process(img)
        img_h, img_w, img_c = img.shape
        # if number of detected faces is 1 (to execlude the data with more than 1 face but only one label)
        faces_features = []
        if results.multi_face_landmarks:
            for face_landmarks  in results.multi_face_landmarks:
                # Get the 468 points.
                features = np.array([
                    [
                        int(lm.x * img_w),
                        int(lm.y * img_h)
                    ]
                    for lm in face_landmarks.landmark
                ], dtype=np.float64)

                features = features.flatten().reshape(1, -1)
                faces_features.append(features)
        return faces_features


    def get_data(self):
        all_imgs = glob(str(self.data_dir / '*.jpg'))
        data = []
        for img in all_imgs:
            img_name = Path(img).stem
            img = cv2.imread(img)
            faces = self.extract_features_from_img(img)
            if faces and len(faces) == 1:
                features = faces[0]    
                mat = sio.loadmat(str(self.data_dir / f'{img_name}.mat'))
                labels = self._get_labels(mat)
                row = np.concatenate((features, labels), axis=1)
                # print('row.shape =', row.shape)
                data.append((img_name, row))
        return data
        

    def get_data_df(self, data):
        n = (data[0][1].shape[1] - 3) // 2

        x_cols_names = [f'x{i}' for i in range(n)]
        y_cols_names = [f'y{i}' for i in range(n)]
        
        features_cols_names = []
        for i in range(len(x_cols_names)):
            features_cols_names.append(x_cols_names[i])
            features_cols_names.append(y_cols_names[i])
        
        labels_cols_names = ['pitch', 'yaw', 'roll']
        cols = features_cols_names + labels_cols_names

        result = pd.DataFrame(
            index=[d[0] for d in data],
            data=np.squeeze([d[1] for d in data]),
            columns=cols,
        ).sort_index()

        return result
        

if __name__ == '__main__':
    path = '/home/ahmad/projects/python/head_pose_estimation/AFLW2000-3D/AFLW2000'
    extractor = DataExtractor(path)

    # Init a cleaner object.
    df = extractor.get_data_df()
    df.to_csv('data.csv')
