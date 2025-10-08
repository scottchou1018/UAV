import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID

CHESSBOARD_CORNERS = (6, 9)
objp = np.zeros((CHESSBOARD_CORNERS[0] * CHESSBOARD_CORNERS[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHESSBOARD_CORNERS[0], 0:CHESSBOARD_CORNERS[1]].T.reshape(-1,2)

# def calibration():
#     cap = cv2.VideoCapture(0)
#     img_points = []
#     obj_points = []

#     while True:
#         ret, frame = cap.read()

#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         ret, corner = cv2.findChessboardCorners(gray_frame, (6, 9), None)
#         if not ret:
#             continue
        
#         img_point = cv2.cornerSubPix(gray_frame, corner, (11, 11), (-1, -1),
#                          (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
#         img_points.append(img_point)
#         obj_points.append(objp)
#         frame = cv2.drawChessboardCorners(frame, (6, 9), img_point, ret)
#         cv2.imshow('frame', frame)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#         if len(img_points) > 4:
#             break
    
#     cap.release()

#     ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray_frame.shape[::-1], None, None)
#     f = cv2.FileStorage('calibration.xml', cv2.FILE_STORAGE_WRITE)
#     f.write('camera_matrix', mtx)
#     f.write('dist_coeff', dist)
#     f.release()
#     print(mtx)

def main():

    fs = cv2.FileStorage('calibration.xml', cv2.FILE_STORAGE_READ)
    intrinsic = fs.getNode("camera_matrix").mat()
    distortion = fs.getNode('dist_coeff').mat()

    print(intrinsic.shape)

    # Tello
    drone = Tello()
    drone.connect()
    #time.sleep(10)
    drone.streamon()
    frame_read = drone.get_frame_read()


    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    while True:
        frame = frame_read.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        markerCorners, markerIds, rejectedCandidates = \
        cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        
        rvec, tvec, _objPoints = \
        cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distortion)
        if markerIds is not None:
            print(tvec.shape)
            for i in range(len(markerIds)):
                # 使用 cv2.drawFrameAxes 繪製座標軸
                # 參數：影像, 相機矩陣, 畸變係數, 旋轉向量, 平移向量, 軸長度(單位與marker_length一致)
                frame = cv2.drawFrameAxes(frame, intrinsic, distortion, rvec[i], tvec[i], 10)
                top_left_corner = markerCorners[i][0][0]
                text_position = (int(top_left_corner[0]), int(top_left_corner[1] - 15))
                cv2.putText(frame, f'x: {tvec[i][0][0]:.2f}, y: {tvec[i][0][1]:.2f}, z: {tvec[i][0][2]:.2f}', text_position, 
                            cv2.FONT_HERSHEY_SIMPLEX, # 字型
                0.5,                      # 字型大小
                (0, 255, 0),              # 顏色 (B, G, R)，這裡是綠色
                2,                        # 線條粗細
                cv2.LINE_AA )

        cv2.imshow("drone", frame)
        key = cv2.waitKey(33)
    
    #cv2.destroyAllWindows()



if __name__ == '__main__':
    main()

