from math import gamma
import matplotlib.pyplot as pit
from sklearn import  datasets
from  sklearn import  svm
digits = datasets.load_digits()
clf = svm.SVC(gamma= 0.001 , C=100)
print(len(digits.data))
x,y = digits.data[:-10],digits.target[:-10]
clf.fit(x,y)
print("Prediction: ",clf.predict(digits.data[-5]))

pit.imshow(digits.images[-5] , cmap=pit.cm.gray_r, interpolation="nearest")
pit.show()
