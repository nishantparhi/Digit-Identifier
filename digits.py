import cv2
import numpy as np

digits=cv2.imread("digits.png",cv2.IMREAD_GRAYSCALE)
test_digits=cv2.imread("test.png",cv2.IMREAD_GRAYSCALE)

test_digits=np.vsplit(test_digits,50)


test_cells=[]
for d in test_digits:
    d=d.flatten()
    test_cells.append(d)
test_cells=np.array(test_cells,dtype=np.float32)
rows=np.vsplit(digits,50)

cells=[]

for row in rows:
    elem=np.hsplit(row,50)
    for i in elem:
        i=i.flatten()
        cells.append(i)

# Conversion np array
cells=np.array(cells,dtype=np.float32)

k=np.arange(10)
cells_labels=np.repeat(k,250)

# Load KNN

knn=cv2.ml.KNearest_create()
knn.train(cells,cv2.ml.ROW_SAMPLE,cells_labels)

ret,result,neighbours,dist=knn.findNearest(test_cells,k=3)

# K value is then nbr of nearest neighbours that we need to find; K can be any 'odd' value

print(result)
