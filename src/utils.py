import imutils
from imutils import face_utils
import cv2
import numpy as np
import scipy.io as sio
import cv2
from math import pi
import dlib
from glob import glob
from pathlib import Path
import pandas as pd


class DataCleaner:
    def __init__(self, path, points_pick='68pts'):
        self.data_dir = Path(path)
        self.points_pick = points_pick
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            '/home/ahmad/projects/python/head_pose_estimation/data')

    def _get_6_points(self, marks):
        return [
            marks[30],     # Nose tip
            marks[8],      # Chin
            marks[36],     # Left eye left corner
            marks[45],     # Right eye right corne
            marks[48],     # Left Mouth corner
            marks[54],     # Right mouth corner
        ]

    def _get_21_points(self, marks):
            left_eye_brew = (17, 19, 21)
            right_eye_brew = (22, 24, 26)
            left_eye = (36, 39)
            right_eye = (42, 45)
            nose = (31, 30, 35)
            lips = (48, 66, 54)
            chin = (8,)
            ears = (2, 14)
            return [marks[i] for i in left_eye_brew] + \
                [marks[i] for i in right_eye_brew] + \
                [marks[i] for i in left_eye] + \
                [marks[i] for i in right_eye] + \
                [marks[i] for i in nose] + \
                [marks[i] for i in lips] + \
                [marks[i] for i in chin] + \
                [marks[i] for i in ears]

    def _get_features(self, marks):
        # Normalization Step:
        # devide by the nose point to normalize the distance from face to the camera.
        # consider the nose point as the center to avoid the change
        # of faces position in images.
        # print('marks =', marks)
        nose_point = marks[30]
        marks = marks - nose_point
        # print('marks after centering =', marks)
        d = np.sqrt(np.sum((marks[30] - marks[8])**2)) # distance from nose to chin
        # print('d =', d)
        marks = marks / d
        # print('marks after scaling =', marks)
        

        # Decide the points selection method.
        if self.points_pick == '68pts':
            features = marks
        elif self.points_pick == '21pts':
            features = self._get_21_points(marks)
        elif self.points_pick == '6pts':
            features = self._get_6_points(marks)
        else:
            raise ValueError(f"points_pick = {self.points_pick} is not a vaild option. please choose '6pts', '21pts' or '68pts'")        
        
        # print('nose_point =', nose_point)
        # print('features =', features)
        features = np.array(features).flatten(order='F')
        return features.reshape(1, -1)

    # def _get_cleaned_features(self, features)
    def _get_labels(self, mat):
        pose = mat.get('Pose_Para')[0, :3]
        labels = pose * 180.0 / pi
        return labels.reshape(1, -1)

    # def get_cleaned_df(self):
    #     all_imgs = glob(str(self.data_dir / '*.jpg'))
    #     data = []
    #     indices = []
    #     print('len(all_imgs) =', len(all_imgs))
    #     i = 1
    #     for img in all_imgs:
    #         img_name = Path(img).stem
    #         print('i =', i, 'img =', img_name)
    #         i += 1
    #         image = cv2.imread(img)
    #         # image = imutils.resize(image, width=500)
    #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #         # detect faces in the grayscale image
    #         rects = self.detector(gray, 1)
    #         # loop over the face detections
    #         if rects:
    #             print('face identified')
    #             indices.append(img_name)
    #             rect = rects[0]
    #             # determine the facial landmarks for the face region, then
    #             # convert the facial landmark (x, y)-coordinates to a NumPy
    #             # array
    #             shape = self.predictor(gray, rect)
    #             shape = face_utils.shape_to_np(shape)
    #             features = self._get_features(shape)

    #             mat = sio.loadmat(str(self.data_dir / f'{img_name}.mat'))
    #             labels = self._get_labels(mat)
    #             data.append((features, labels))

    #     data = np.squeeze([np.concatenate((arr[0], arr[1]), axis=1)
    #                       for arr in data])
    #     n = (data[0].shape[0]-3) // 2
    #     cols = [f'x{i}' for i in range(
    #         1, n+1)] + [f'y{i}' for i in range(1, n+1)] + ['pitch', 'yaw', 'roll']

    #     return pd.DataFrame(
    #         index=indices,
    #         data=data,
    #         columns=cols,
    #     ).sort_index()

    def get_cleaned_df(self):
        all_imgs = glob(str(self.data_dir / '*.mat'))
        data = []
        indices = []
        print('len(all_imgs) =', len(all_imgs))
        i = 1
        for img in all_imgs: # [str(self.data_dir / 'image01152.mat')]:
            img_name = Path(img).stem
            print('i =', i, 'img =', img_name)
            i += 1
            
            mat = sio.loadmat(img)
            indices.append(img_name)
            features = self._get_features(mat.get('pt3d_68')[:2].T)
            

            labels = self._get_labels(mat)
            data.append((features, labels))

        data = np.squeeze([np.concatenate((arr[0], arr[1]), axis=1)
                          for arr in data])
        n = (data[0].shape[0]-3) // 2
        cols = [f'x{i}' for i in range(1, n+1)] \
             + [f'y{i}' for i in range(1, n+1)] \
             + ['pitch', 'yaw', 'roll']

        result = pd.DataFrame(
            index=indices,
            data=data,
            columns=cols,
        ).sort_index()

        # drop the zero columns after centering around them.
        cols = [c for c in result.columns if len(result[c].unique()) == 1]
        return result.drop(columns=cols)


if __name__ == '__main__':
    # Init a cleaner object.
    cleaner = DataCleaner(
        '/home/ahmad/projects/python/head_pose_estimation/AFLW2000-3D/AFLW2000')
    
    # Get and save a dataframe (6 points as features).
    cleaner.points_pick = '6pts'
    df = cleaner.get_cleaned_df()
    df.to_csv('cleaned_data_6pts.csv')

    # Get and save a dataframe (21 points as features).
    cleaner.points_pick = '21pts'
    df = cleaner.get_cleaned_df()
    df.to_csv('cleaned_data_21pts.csv')

    # Get and save a dataframe (69 points as features).
    cleaner.points_pick = '68pts'
    df = cleaner.get_cleaned_df()
    df.to_csv('cleaned_data_68pts.csv')
