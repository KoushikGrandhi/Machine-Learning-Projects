import cv2
import numpy as np

tracking_points = []
detection_points_count = 9
x1 = None
y1 = None
x2 = None
y2 = None
object_hist=None

def rescaling(frame, wpercent=130, hpercent=130):
    w = int(frame.shape[1] * wpercent / 100)
    h = int(frame.shape[0] * hpercent / 100)
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

def draw_detection_areas(frame):
    rows, cols, _ = frame.shape
    global detection_points_count, x1, y1, x2, y2
    
    x1 = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
         12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

    y1 = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
         10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

    x2 = x1 + 10
    y2 = y1 + 10
    for i in range(detection_points_count):
        cv2.rectangle(frame, (y1[i], x1[i]),
                      (y2[i], x2[i]),
                      (255, 0, 0), 2)

    return frame


def object_histogram(frame):
    global x1, y1

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    region = np.zeros([90, 10, 3], dtype=hsv_frame.dtype) #masking Region of interest for extracting histogram

    for i in range(detection_points_count):
        region[i * 10: i * 10 + 10, 0: 10] = hsv_frame[x1[i]:x1[i] + 10,
                                          y1[i]:y1[i] + 10]

    object_hist = cv2.calcHist([region], [0, 1], None, [180, 256], [0, 180, 0, 256])
    print("Object histogram read complete")
    return cv2.normalize(object_hist, object_hist, 0, 255, cv2.NORM_MINMAX)

# Masking done using contours created based on object_histogram
def object_mask(frame, hist):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    dst = cv2.calcBackProject([hsv_frame], [0, 1], hist, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))#kernel can be changed
    cv2.filter2D(dst, -1, disc, dst)

    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)

    thresh = cv2.merge((thresh, thresh, thresh))
    temp_mask_image= cv2.bitwise_and(frame, thresh)
    temp_mask_image = cv2.erode(temp_mask_image, None, iterations=2)
    temp_mask_image = cv2.dilate(temp_mask_image, None, iterations=2)
    gray_masked_image = cv2.cvtColor(temp_mask_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_masked_image, 0, 255, 0)
    cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cont
    


def centroid(max_contour):
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None



def farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        c_x, c_y = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)
        # Euclidian distance
        x_dist = cv2.pow(cv2.subtract(x, c_x), 2)
        y_dist = cv2.pow(cv2.subtract(y, c_y), 2)
        dist = cv2.sqrt(cv2.add(x_dist, y_dist))

        max_dist = np.argmax(dist)

        if max_dist < len(s):
            farthest_defect = s[max_dist]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None


def show_tracks(frame, tracking_points):
    if tracking_points is not None:
        
        for i in range(len(tracking_points)):
            temp= list(tracking_points[i])
            frame= cv2.line(frame, (temp[0],temp[1]), (temp[0]+2,temp[1]+2), [0, 255, 255], 2)


def main_frame(frame, object_hist): 

    contours_list = object_mask(frame, object_hist)
    max_contour = max(contours_list, key=cv2.contourArea)

    centroid_point = centroid(max_contour)
    cv2.circle(frame, centroid_point, 5, [255, 0, 255], -1)
    print("In main_frame method max_cont : ", max_contour)
    if max_contour is not None:
        chull = cv2.convexHull(max_contour, returnPoints=False)
        cdefects = cv2.convexityDefects(max_contour, chull)
        farthest_pt = farthest_point(cdefects, max_contour, centroid_point)
        cv2.circle(frame, farthest_pt, 5, [0, 0, 255], -1)
        if len(tracking_points) < 20:
            tracking_points.append(centroid_point) #farthest_point can also be used for finger tracking
        else:
            tracking_points.pop(0)
            tracking_points.append(centroid_point)
        print('tracking_points: ',tracking_points)
        show_tracks(frame, tracking_points)


def main():
    global object_hist
    object_hist_flag = False
    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        pressed_key = cv2.waitKey(1)
        # print(str(pressed_key))
        _,frame = capture.read()
        frame = cv2.flip(frame, 1)
        # cv2.imwrite("Press g to start.", frame)
        if pressed_key & 0xFF == ord('g'): # Pressing g will generate the histogram of the object
            object_hist_flag = True
            object_hist = object_histogram(frame)

        if object_hist_flag:
            main_frame(frame, object_hist)

        else:
            frame = draw_detection_areas(frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,'Press g to start',(50, 50),font, 1, (0, 255, 255), 2,cv2.LINE_4)
        cv2.imshow("Video", rescaling(frame))
        if pressed_key == 27:
            break

    cv2.destroyAllWindows()
    capture.release()


if __name__ == '__main__':
    main()