from keyboard_djitellopy import keyboard

import cv2
import numpy as np
from pyimagesearch.pid import PID
import math
MAX_SPEED = 50
MARKER_NUM = 2

z_pid = PID(kP = 5, kI= 0, kD=1)
y_pid = PID(kP = 5, kI= 0, kD=1)
x_pid = PID(kP = 5, kI= 0, kD=1)
yaw_pid = PID(kP = 10, kI= 1, kD=1)

def filter_marker_pos(marker_id: int, markerIds, tvec, rvec):
    for i in range(len(markerIds)):
        if markerIds[i] == marker_id:
            return (tvec[i], rvec[i])
    return (None, None)


def to_target(drone, marker_id, to_x, to_y, to_z, to_deg, markerIds, tvec_all, rvec_all,):
    tvec, rvec = filter_marker_pos(marker_id, markerIds, tvec_all, rvec_all)
    if tvec is None:
        return False
    z_update = tvec[0, 2] - to_z
    z_update = z_pid.update(z_update, sleep = 0)
    z_update = min(z_update, MAX_SPEED)
    z_update = max(z_update, -MAX_SPEED)


    x_update = tvec[0, 0] - to_x
    x_update = x_pid.update(x_update, sleep = 0)
    x_update = min(x_update, MAX_SPEED)
    x_update = max(x_update, -MAX_SPEED)

    y_update = -tvec[0, 1] - to_y
    y_update = y_pid.update(y_update, sleep = 0)
    y_update = min(y_update, MAX_SPEED)
    y_update = max(y_update, -MAX_SPEED)
    
    R_mat, _ = cv2.Rodrigues(rvec)
    z_unit = np.array([[0], [0], [1]])
    z_rot = np.matmul(R_mat, z_unit)
    z_rot[1][0] = 0

    ang = math.atan2(z_rot[2][0], z_rot[0][0])
    deg = -math.degrees(ang)

    yaw_update = deg - 90 - to_deg
    yaw_update = y_pid.update(yaw_update, sleep = 0)
    yaw_update = min(yaw_update, MAX_SPEED)
    yaw_update = max(yaw_update, -MAX_SPEED)
    drone.send_rc_control(int(x_update // 2), int(z_update // 2), int(y_update // 2), int(yaw_update // 2))
    return True

if __name__ == "__main__":
    from djitellopy import Tello

    fs = cv2.FileStorage('calibration.xml', cv2.FILE_STORAGE_READ)
    intrinsic = fs.getNode("camera_matrix").mat()
    distortion = fs.getNode('dist_coeff').mat()

    drone = Tello()
    drone.connect()
    drone.streamon()
    frame_read = drone.get_frame_read()

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    

    z_pid.initialize()
    y_pid.initialize()
    x_pid.initialize()
    yaw_pid.initialize()

    stage = 1


    done_markers = [False for _ in range(MARKER_NUM + 1)]

    while True:
        frame = frame_read.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        markerCorners, markerIds, rejectedCandidates = \
        cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        text_position = (int(10), int(10))
        cv2.putText(frame, f'battery: {drone.get_battery()}', text_position, 
                    cv2.FONT_HERSHEY_SIMPLEX, # 字型
        0.5,                      # 字型大小
        (0, 255, 0),              # 顏色 (B, G, R)，這裡是綠色
        2,                        # 線條粗細
        cv2.LINE_AA )
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
        

        
        key = cv2.waitKey(1)
        if key != -1:
            keyboard(drone, key)
            continue
        
        

        if tvec is None:
            drone.send_rc_control(0, 0, 0, 0)
            continue
        
        
        id1_tvec, id1_rvec = filter_marker_pos(1, markerIds, tvec, rvec)
        id2_tvec, id2_rvec = filter_marker_pos(2, markerIds, tvec, rvec)

        if id1_tvec is not None:
            to_target(drone, 1, 0, 0, 100, 0,markerIds, tvec, rvec)


        # if(id1_tvec is None and id2_tvec is None):
        #     if stage == 1:
        #         drone.send_rc_control(0, 0, 1, 0)
        #     continue
        
        # follow_id = 1 if id1_tvec is not None else 2

        # if(id1_tvec is not None and id2_tvec is not None):
        #     follow_id = 1 if id1_tvec[2] > id2_tvec[2] else 2

        # if(id1_tvec is None):
        #     continue
        
        # if not done_markers[follow_id]:
            
        #     to_target(drone, id1_tvec, id1_rvec, 0, 0, 100, 0)

        # if follow_id == 1:
        #     to_target(drone, id1_tvec, id1_rvec, 0, 0, 100, 0)
        # elif follow_id == 2:



        # z_update = tvec[0,0,2] - 100
        # z_update = z_pid.update(z_update, sleep = 0)
        # z_update = min(z_update, MAX_SPEED)
        # z_update = max(z_update, -MAX_SPEED)


        # x_update = tvec[0,0,0]
        # x_update = x_pid.update(x_update, sleep = 0)
        # x_update = min(x_update, MAX_SPEED)
        # x_update = max(x_update, -MAX_SPEED)

        # y_update = -tvec[0,0,1]
        # y_update = y_pid.update(y_update, sleep = 0)
        # y_update = min(y_update, MAX_SPEED)
        # y_update = max(y_update, -MAX_SPEED)

        # R_mat, _ = cv2.Rodrigues(rvec[0])
        # z_unit = np.array([[0], [0], [1]])
        # z_rot = np.matmul(R_mat, z_unit)
        # z_rot[1][0] = 0

        # ang = math.atan2(z_rot[2][0], z_rot[0][0])
        # deg = math.degrees(ang)

        # yaw_update = -deg - 90
        # yaw_update = y_pid.update(yaw_update, sleep = 0)
        # yaw_update = min(yaw_update, MAX_SPEED)
        # yaw_update = max(yaw_update, -MAX_SPEED)



        # drone.send_rc_control(int(x_update // 2), int(z_update // 2), int(y_update // 2), int(yaw_update // 2))



    cv2.destroyAllWindows()
    drone.streamoff()
    drone.end()