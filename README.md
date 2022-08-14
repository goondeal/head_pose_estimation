
# Head-Pose-Estimation

- Notice that there are 3 branches to try different apploaches.
- Last updates are in media_pipe branch.

Video: https://drive.google.com/drive/folders/19ERe7SHhE7KHihkgOamZdD3l0HtV1si0


This is a simple project with a target to estimate the head pose **(pitch, yaw, roll)** from 2-D face landmarks data using **AFLAW2000** dataset.

![image](https://user-images.githubusercontent.com/71623159/174402203-15a79be9-1798-4c44-b120-0d5b1a0e8bc2.png)


### Approach

1. Use [mediapipe](https://google.github.io/mediapipe/) to detect faces 
    and extract features from the image (facial landmarks as 468 2D points),
    for each image in the dataset.
2. Carry out a dimensionality reduction as a feature selection to reduce model complexity.
3. Split the data to [0.80, 0.20] for training set, validation set respectivly. 
    I choosed not to have a test set beacause:
    1. We do not have much data (only 2000 images).
    2. Testing will be online on real video.
4. Try different intuitive models according to the problem and the data.
5. Evaluation metric is MSE using `r2_score` from sklearn.
6. Regarding code: I used an OOP paradigm to be easily to test and maintain.
---


### Challenges:

1. The same person's face can appear anywhere in an image, yet it is the 
    same face but it will result a different features eachtime its place
    changes in image. 
    - To overcome that, I picked a fixed point (nose point) and considered it the origin of all landmarks (mean normalization).

2. The same person's face can appear so close or far away from the camera, yet it is the 
    same face but it will result a different features the distance between his face and the camera changes. 
    - To overcome that, I scaled all points with a distance between 2 fixed points (nose & chin)

