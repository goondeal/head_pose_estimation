import cv2
import numpy as np
import scipy.io as sio
from math import pi
import mediapipe as mp
from glob import glob
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


class DataCleaner:
    def __init__(self, path, points_pick='68pts'):
        self.data_dir = Path(path)
        self.points_pick = points_pick
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=2)


    def get_features(self, marks):
        # Normalization Step:
        # devide by the nose point to normalize the distance from face to the camera.
        # consider the nose point as the center to avoid the change
        # of faces position in images.
        # print('marks =', marks)
        nose_point = marks[1]
        marks = marks - nose_point
        # print('marks after centering =', marks)
        d = np.sqrt(np.sum((marks[377] - marks[1])**2)) # distance from nose to chin
        # print('d =', d)
        marks = marks / d
        features = marks.flatten(order='F')
        return features

    # def _get_cleaned_features(self, features)
    def _get_labels(self, mat):
        pose = mat.get('Pose_Para')[0, :3]
        labels = pose # * 180.0 / pi
        return labels

    def get_cleaned_df(self):
        all_imgs = glob(str(self.data_dir / '*.jpg'))
        data = []
        indices = []
        print('len(all_imgs) =', len(all_imgs))
        for img in all_imgs:
            img_name = Path(img).stem
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Get the result
            results = self.face_mesh.process(img)
            # Convert the color space from RGB to BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_h, img_w, img_c = img.shape
            
            if results.multi_face_landmarks and len(results.multi_face_landmarks) == 1:
                # print('results.multi_face_landmarks =', len(results.multi_face_landmarks))
                for face_landmarks in results.multi_face_landmarks:
                    indices.append(img_name)
                    # print(face_landmarks.landmark[0])
                    features = np.array([
                        [
                            int(lm.x * img_w),
                            int(lm.y * img_h)
                        ]
                        for lm in face_landmarks.landmark 
                    ], dtype=np.float64)

                    features = self.get_features(features)

                    mat = sio.loadmat(str(self.data_dir / f'{img_name}.mat'))
                    labels = self._get_labels(mat)
                    data.append((features, labels))
        #         cv2.imshow(img_name, img)
        # cv2.waitKey(0)
        features = np.array([d[0] for d in data])
        # print('features shape =', features.shape)
        features = self.apply_pca(features)
        features = MinMaxScaler().fit_transform(features)

        data = np.concatenate((features, [d[1] for d in data]), axis=1)
        print(data.shape)
        n = data.shape[1]-3
        cols = [f'x{i}' for i in range(1, n+1)] \
             + ['pitch', 'yaw', 'roll']

        result = pd.DataFrame(
            index=indices,
            data=data,
            columns=cols,
        ).sort_index()

        # drop the zero columns after centering around them.
        # cols = [c for c in result.columns if len(result[c].unique()) == 1]
        return result


    def apply_pca(self, data):
        pca = PCA(n_components=8)
        new_data = pca.fit_transform(data)
        print('variance ratio =', pca.explained_variance_ratio_.sum())
        return new_data

    # def get_cleaned_df(self):
    #     all_imgs = glob(str(self.data_dir / '*.mat'))
    #     data = []
    #     indices = []
    #     print('len(all_imgs) =', len(all_imgs))
    #     i = 1
    #     for img in all_imgs: # [str(self.data_dir / 'image01152.mat')]:
    #         img_name = Path(img).stem
    #         print('i =', i, 'img =', img_name)
    #         i += 1
            
    #         mat = sio.loadmat(img)
    #         indices.append(img_name)
    #         features = self._get_features(mat.get('pt3d_68')[:2].T)
            

    #         labels = self._get_labels(mat)
    #         data.append((features, labels))

    #     data = np.squeeze([np.concatenate((arr[0], arr[1]), axis=1)
    #                       for arr in data])
    #     n = (data[0].shape[0]-3) // 2
    #     cols = [f'x{i}' for i in range(1, n+1)] \
    #          + [f'y{i}' for i in range(1, n+1)] \
    #          + ['pitch', 'yaw', 'roll']

    #     result = pd.DataFrame(
    #         index=indices,
    #         data=data,
    #         columns=cols,
    #     ).sort_index()

        # drop the zero columns after centering around them.
        # cols = [c for c in result.columns if len(result[c].unique()) == 1]
        # return result.drop(columns=cols)


if __name__ == '__main__':
    # Init a cleaner object.
    cleaner = DataCleaner(
        '/home/ahmad/projects/python/head_pose_estimation/AFLW2000-3D/AFLW2000')
    
    # Get and save a dataframe (6 points as features).
    # cleaner.points_pick = '6pts'
    # df = cleaner.get_cleaned_df()
    # df.to_csv('cleaned_data_6pts.csv')

    # Get and save a dataframe (21 points as features).
    # cleaner.points_pick = '21pts'
    # df = cleaner.get_cleaned_df()
    # df.to_csv('cleaned_data_21pts.csv')

    # Get and save a dataframe (69 points as features).
    cleaner.points_pick = '68pts'
    df = cleaner.get_cleaned_df()
    df.to_csv('cleaned_data_mp.csv')
