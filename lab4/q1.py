import cv2
import numpy as np

CHESSBOARD_CORNERS = (6, 9)
objp = np.zeros((CHESSBOARD_CORNERS[0] * CHESSBOARD_CORNERS[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHESSBOARD_CORNERS[0], 0:CHESSBOARD_CORNERS[1]].T.reshape(-1,2)
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    img_points = []
    obj_points = []
    
    while True:
        ret, frame = cap.read()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corner = cv2.findChessboardCorners(gray_frame, (6, 9), None)
        if not ret:
            continue
        
        img_point = cv2.cornerSubPix(gray_frame, corner, (11, 11), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
        img_points.append(img_point)
        obj_points.append(objp)
        frame = cv2.drawChessboardCorners(frame, (6, 9), img_point, ret)
        cv2.imshow('frame', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if len(img_points) > 4:
            break
    
    cap.release()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray_frame.shape[::-1], None, None)
    f = cv2.FileStorage('calibration.xml', cv2.FILE_STORAGE_WRITE)
    f.write('camera_matrix', mtx)
    f.write('dist_coeff', dist)
    f.release()
    print(mtx)

