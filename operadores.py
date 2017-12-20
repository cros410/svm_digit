# Import the modules
import os
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
from collections import Counter
from sklearn_porter import Porter
from sklearn.model_selection import train_test_split
import cv2

def appendData(data, lab , tipo):
    cont = 0
    hog_list = []
    for filename in os.listdir("D:\\Christian-Data\\Proyectos\\Python\\digitRecognition\\numeros\\" + str(tipo)):
        cont = cont + 1
        im = cv2.imread("D:\\Christian-Data\\Proyectos\\Python\\digitRecognition\\numeros\\" + str(tipo) +"\\"+ filename)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
        ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY)
        image, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]
        rect = rects[0]
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        hog_list.append(roi_hog_fd)
    a_l = [tipo for x in range(cont)]
    np_l = np.array(a_l,'int')
    np_f = np.array(hog_list,'float64')
    l = np.append(lab, np_l)
    f = np.append(data , np_f, axis=0)
    return l , f


label, hog_features = appendData(val)

print ("Features final:" + str(len(hog_features)))
print ("Elementos final :" + str(labels.size))


X_train, X_test, y_train, y_test = train_test_split( hog_features, labels, test_size=0.3, random_state=0)
clf = LinearSVC()
clf.fit(X_train, y_train)
print("Porcentaje : " + str(clf.score(X_test, y_test)))
porter = Porter(clf, language='java')
output = porter.export(embedded=True)
file = open("svm.java", "w")
file.write(output)
#print(output)
# Save the classifier
#joblib.dump(clf, "digits_cls.pkl", compress=3)


