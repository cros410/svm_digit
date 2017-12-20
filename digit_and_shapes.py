#cargar paquetes
import cv2
import numpy as np
import argparse
import imutils
from pyimagesearch.shapedetector import ShapeDetector
from sklearn.externals import joblib
from skimage.feature import hog

#cargar imagen
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())
clf = joblib.load("digits_cls.pkl")
image = cv2.imread(args["image"])
cv2.imshow("Image", image)
cv2.waitKey(0)
#separar imagenes
boundaries = [
	([0, 0, 70], [97, 100, 255]),#ROJO [0, 0, 70], [97, 100, 255]
    ([0, 70, 0], [250, 255, 255]) #VERDE [0, 70, 0], [250, 255, 255]
]
images = [0,0]
count = 0
for (lower, upper) in boundaries:
    # crear NumPy de los rangos
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")
        
    # encontrar el color con el espefico rango
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask = mask)
    images[count] = np.hstack([output])
    count=count+1

shape_f = images[0]
number_f = images[1]
cv2.imshow("Image", shape_f)
cv2.waitKey(0)
cv2.imshow("Image", number_f)
cv2.waitKey(0)

# RECONOCER LA FORMA
#redimencionar la imagen
resized = imutils.resize(shape_f, width=200)
ratio = shape_f.shape[0] / float(resized.shape[0])

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
ret, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
sd = ShapeDetector()

for c in cnts:
    # compute the center of the contour, then detect the name of the
    # shape using only the contour
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    shape = sd.detect(c)

    # multiply the contour (x, y)-coordinates by the resize ratio,
    # then draw the contours and the name of the shape on the image
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    cv2.drawContours(shape_f, [c], -1, (0, 255, 0), 2)
    cv2.putText(shape_f, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,0.5, (204, 153, 3), 2)

    # show the output image
    cv2.imshow("Image", shape_f)
    cv2.waitKey(0)

# RECONOCER EL NUMERO

im_gray_d = cv2.cvtColor(number_f, cv2.COLOR_BGR2GRAY)
im_gray_d = cv2.GaussianBlur(im_gray_d, (5, 5), 0)
ret, im_th = cv2.threshold(im_gray_d, 90, 255, cv2.THRESH_BINARY)
image_d, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

for rect in rects:
    # Draw the rectangles
    cv2.rectangle(number_f, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    # Calculate the HOG features
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    print(np.array([roi_hog_fd]))
    nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
    cv2.putText(number_f, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (204, 153, 3), 3)
    

cv2.imshow("Resulting Image with Rectangular ROIs", number_f)
cv2.waitKey(0)