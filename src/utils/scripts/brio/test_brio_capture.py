import numpy as np
import cv2
import time

cap = cv2.VideoCapture(4)

cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # turn the autofocus off
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

BRIO_INTR = np.asarray([[3.34368560e+03, 0.00000000e+00, 1.80619760e+03],
 [0.00000000e+00, 3.34161191e+03, 1.09037428e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

BRIO_DIST = np.asarray([ 0.22325137, -0.73638229, -0.00355125, -0.0042986,   0.96319653])

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(BRIO_INTR, BRIO_DIST, (width,height), 1, (width,height))
mapx, mapy = cv2.initUndistortRectifyMap(BRIO_INTR, BRIO_DIST, None, newcameramtx, (width,height), 5)
if not cap.isOpened():
 print("Cannot open camera")
 exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here

    # undistort
    dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w] # (2119, 3804, 3)
    # Display the resulting frame
    # print(dst.shape)
    dst = cv2.resize(dst, (dst.shape[1]//3, dst.shape[0]//3))

    cv2.imshow('frame', dst)
    if cv2.waitKey(1) == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()