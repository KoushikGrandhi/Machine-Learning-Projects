import os
import cv2 as cv
import numpy as np

def stackImages_ECC(file_list):
    M = np.eye(3, 3, dtype=np.float32)

    temp_image = None
    final_image_cv = None

    for file in file_list:
        image = cv.imread(file,1).astype(np.float32) / 255
        print(file)
        if temp_image is None:
            # convert to gray scale floating point image
            temp_image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
            final_image_cv = image
        else:
          # Specify the number of iterations.
            number_of_iterations = 5000
            # Specify the threshold of the increment
            # in the correlation coefficient between two iterations
            termination_eps = 1e-10
            # Define termination criteria
            criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
            # Estimate perspective transform
            retval, warpMatrix = cv.findTransformECC(cv.cvtColor(image,cv.COLOR_BGR2GRAY), temp_image, M, cv.MOTION_HOMOGRAPHY,criteria,inputMask=None,gaussFiltSize=1)
            width, height, _ = image.shape
            # Align image to first image
            image = cv.warpPerspective(image, warpMatrix, (height, width))
            final_image_cv += image

    final_image_cv /= len(file_list)
    final_image_cv = (final_image_cv*255).astype(np.uint8)
    return final_image_cv


# Align and stack images by matching ORB keypoints
# Faster but less accurate
def stackImages_ORB(file_list):

    orb = cv.ORB_create()

    # disable OpenCL to because of bug in ORB in OpenCV 3.1
    cv.ocl.setUseOpenCL(False)

    final_image_cv = None
    temp_image = None
    first_kp = None
    first_des = None
    for file in file_list:
        print(file)
        image = cv.imread(file,1)
        image_semi_final = image.astype(np.float32) / 255

        # compute the descriptors with ORB
        kp = orb.detect(image, None)
        kp, des = orb.compute(image, kp)

        # create BFMatcher object
        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        if temp_image is None:
            # Save keypoints for first image
            final_image_cv = image_semi_final
            temp_image = image
            first_kp = kp
            first_des = des
        else:
             # Find matches and sort them in the order of their distance
            matches = matcher.match(first_des, des)
            matches = sorted(matches, key=lambda x: x.distance)

            src_pts = np.float32(
                [first_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate perspective transformation
            M, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)
            w, h, _ = image_semi_final.shape
            image_semi_final = cv.warpPerspective(image_semi_final, M, (h, w))
            final_image_cv += image_semi_final

    final_image_cv /= len(file_list)
    final_image_cv = (final_image_cv*255).astype(np.uint8)
    return final_image_cv

if __name__ == '__main__':

    image_folder= "input"
    if not os.path.exists(image_folder):
        print("Invalid path")
        exit()

    file_list = sorted(os.listdir(image_folder))
    file_list = [os.path.join(image_folder, x)
                 for x in file_list if x.endswith(('.jpeg', '.png','.bmp','.JPG'))]
    method= "ORB"
    # method= "ECC"

    if method == 'ECC':
        # Enhanced Correlation Coefficient (ECC) Maximization
        final_image_cv = stackImages_ECC(file_list)

    elif method == 'ORB':
        # ORB point mapping method
        final_image_cv = stackImages_ORB(file_list)

    else:
        print("Method not valid")
        exit()

    # cv2.imwrite('output_image',final_image_cv)
    cv.imshow(final_image_cv)
    cv.waitKey(0)


