import numpy as np
import cv2
import matplotlib.pyplot as plt

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


print(width, height)
if not cap.isOpened():
 print("Cannot open camera")
 exit()

# Capture frame-by-frame
ret, frame = cap.read()

# if frame is read correctly ret is True
if not ret:
    print("Can't receive frame (stream end?). Exiting ...")
    # break
# Our operations on the frame come here

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(BRIO_INTR, BRIO_DIST, (width,height), 1, (width,height))
# undistort
dst = cv2.undistort(frame, BRIO_INTR, BRIO_DIST, None, newcameramtx)
 
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

print(dst.shape)
print(newcameramtx)
rgb_frame = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# Display the resulting frame
# cv2.imshow('frame', frame)
# if cv2.waitKey(1) == ord('q'):
#     break

cy = newcameramtx[1,2]
cx = newcameramtx[0,2]
print(cy)
print(cx)

cy_line_x = [0, width]
cy_line_y = [cy, cy]
cx_line_x = [cx, cx]
cx_line_y = [0, height]
plt.plot(cy_line_x, cy_line_y, 'r')
plt.plot(cx_line_x, cx_line_y, 'r')
plt.imshow(rgb_frame)
plt.show()
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()