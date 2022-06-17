from cgitb import grey
import cv2
import dlib
import imutils
from imutils import face_utils
import mediapipe as mp
import numpy as np
import time
from math import sin, cos

from sympy import true

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/ahmad/projects/python/head_pose_estimation/data')

# cap = cv2.VideoCapture(0)
# points = {17, 19, 21, 22, 24, 26, 36, 39, 42, 45, 31, 30, 35, 48, 66, 54, 8, 2, 14}

# while True:
#     success, image = cap.read()


#     # Flip the image horizontally for a later selfie-view display
#     # Also convert the color space from BGR to RGB
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # To improve performance
#     # image.flags.writeable = False

#     # Get the result
#     rects = detector(image, 1)
#     # loop over the face detections
#     for rect in rects:
        
#         # determine the facial landmarks for the face region, then
#         # convert the facial landmark (x, y)-coordinates to a NumPy
#         # array
#         shape = predictor(image, rect)
#         for n in points: # range(1, 68):
#             x = shape.part(n).x
#             y = shape.part(n).y
#             cv2.circle(image, (x, y), 1, (0, 255, 255), 1)

#     cv2.imshow("Face Landmarks", image)
        


#     # To improve performance
#     # image.flags.writeable = True

#     # Convert the color space from RGB to BGR
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)



#     key= 0xFF & cv2.waitKey(1)
#     if key & 0xFF == 27: #Exit program when the user presses 'esc'
#         break
    
# cv2.destroyAllWindows()
# cap.release()


# Set up some required objects
# video_capture = cv2.VideoCapture(0) #Webcam object
# detector = dlib.get_frontal_face_detector() #Face detector
# predictor = dlib.shape_predictor('/home/ahmad/projects/python/head_pose_estimation/data')
# ("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file

# while True:
#     ret, frame = video_capture.read()
#     frame = cv2.resize(frame, (512, 256))#resize

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     clahe_image = clahe.apply(gray)

#     detections = detector(clahe_image, 1) #Detect the faces in the image

#     for k,d in enumerate(detections): #For each detected face
#         shape = predictor(clahe_image, d) #Get coordinates
#         for i in range(1, 68): #There are 68 landmark points on each face
#             cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) #For each point, draw a red circle with thickness2 on the original frame
#     cv2.imshow("origin", frame)        

#     key= 0xFF & cv2.waitKey(1)
#     if key & 0xFF == 27: #Exit program when the user presses 'esc'
#         break

# # When everything done, release the capture
# video_capture.release()
# cv2.destroyAllWindows()








mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    yaw = -yaw
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()

    start = time.time()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False

    # Get the result
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    detections = detector(image, 1) #Detect the faces in the image

    for k,d in enumerate(detections): #For each detected face
        shape = predictor(image, d) #Get coordinates
        for i in range(1, 68): #There are 68 landmark points on each face
            cv2.circle(image, (shape.part(i).x, shape.part(i).y), 1, (255,0,0), thickness=2) #For each point, draw a red circle with thickness2 on the original frame


    landmark_points_68 = [
        162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
        296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
        380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]

    if results.multi_face_landmarks:
        # print('results.multi_face_landmarks =', len(results.multi_face_landmarks))
        for face_landmarks in results.multi_face_landmarks:
            landmarks_extracted = []
            for index in landmark_points_68:
                x = int(face_landmarks.landmark[index].x * img_w)
                y = int(face_landmarks.landmark[index].y * img_h)
                landmarks_extracted.append((x, y))
            # for idx, lm in enumerate(face_landmarks.landmark):
            #     if idx in {33, 263, 1, 61, 291, 199}:
            #         # if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
            #         if idx == 1:
            #             nose_2d = (lm.x * img_w, lm.y * img_h)
            #             nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

            #         x, y = int(lm.x * img_w), int(lm.y * img_h)

            #         # Get the 2D Coordinates
            #         face_2d.append([x, y])

            #         # Get the 3D Coordinates
            #         face_3d.append([x, y, lm.z])

            # # Convert it to the NumPy array
            # face_2d = np.array(face_2d, dtype=np.float64)
            # print('face_2d.shape =', face_2d.shape)
            # print('face_2d =', face_2d)

        #     # Convert it to the NumPy array
        #     face_3d = np.array(face_3d, dtype=np.float64)

        #     # The camera matrix
        #     focal_length = 1 * img_w

        #     cam_matrix = np.array([[focal_length, 0, img_h / 2],
        #                            [0, focal_length, img_w / 2],
        #                            [0, 0, 1]])

        #     # The distortion parameters
        #     dist_matrix = np.zeros((4, 1), dtype=np.float64)

        #     # Solve PnP
        #     success, rot_vec, trans_vec = cv2.solvePnP(
        #         face_3d, face_2d, cam_matrix, dist_matrix)

        #     # Get rotational matrix
        #     rmat, jac = cv2.Rodrigues(rot_vec)

        #     # Get angles
        #     angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        #     # Get the y rotation degree
        #     x = angles[0] * 360
        #     y = angles[1] * 360
        #     z = angles[2] * 360

        #     # See where the user's head tilting
        #     if y < -10:
        #         text = "Looking Left"
        #     elif y > 10:
        #         text = "Looking Right"
        #     elif x < -10:
        #         text = "Looking Down"
        #     elif x > 10:
        #         text = "Looking Up"
        #     else:
        #         text = "Forward"

        #     # Display the nose direction
        #     nose_3d_projection, jacobian = cv2.projectPoints(
        #         nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            
        #     p1 = (int(nose_2d[0]), int(nose_2d[1]))
        #     p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
        #     draw_axis(image, angles[0], angles[1], angles[2], p1[0], p1[1])
        #     cv2.line(image, p1, p2, (255, 255, 255), 3)

        #     # Add the text on the image
        #     cv2.putText(image, text, (20, 50),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        #     cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #     cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #     cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # end = time.time()
        # totalTime = end - start

        # fps = 1 / totalTime
        # #print("FPS: ", fps)

        # cv2.putText(image, f'FPS: {int(fps)}', (20, 450),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        for p in landmarks_extracted: #There are 68 landmark points on each face
            cv2.circle(image, p, 1, (0,0,255), thickness=2) #For each point, draw a red circle with thickness2 on the original frame
        # mp_drawing.draw_landmarks(
        #     image=image,
        #     # landmark_list=face_landmarks,
        #     landmark_list=face_landmarks[landmarks_extracted],

        #     # connections=mp_face_mesh.FACE_CONNECTIONS,
        #     landmark_drawing_spec=drawing_spec,
        #     connection_drawing_spec=drawing_spec,
        # )

    cv2.imshow('Head Pose Estimation', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break
        

cap.release()
