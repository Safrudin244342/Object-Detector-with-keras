from detection_helper import image_pyramid
from detection_helper import sliding_window
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# load model
model = load_model("model/handDetectorTest.keras")

# load image
orig = cv2.imread("img/hand2.jpg")
if orig.shape[0] > 1000 and orig.shape[1] > 1000:
    orig = cv2.resize(orig, (600, 600))

orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
img = orig
(W, H) = orig.shape[:2]

# make image pyramid
pyramid = image_pyramid(orig, scale=1.5, minSize=(50, 50))

rois = []
locs = []

i = 0
for image in pyramid:
    # sliding windows
    scaleImg = W / float(image.shape[1])

    for (x, y, roiOrig) in sliding_window(image, 25, (50, 50)):
        newX = int(x * scaleImg)
        newY = int(y * scaleImg)
        newW = int(50 * scaleImg)
        newH = int(50 * scaleImg)
        box = (newX, newY, newX + newW, newY + newH)
        if box == (225, 112, 337, 224):
            print(scaleImg)

        roi = img_to_array(roiOrig)
        rois.append(roi)
        locs.append(box)

        clone = orig.copy()
        cv2.rectangle(clone, (newX, newY), (newX + newW, newY + newH),
                      (0, 255, 0), 2)

rois = np.array(rois)
rois = [(1. / 255)] * rois[np.newaxis]
rois = rois[0]
preds = model.predict(rois)

objectRaw = []
for i, pred in enumerate(preds):
    if pred[0] < 0.004:
        objectRaw.append(locs[i])

objectLocs = [[0, 0, W, H]]
for object in objectRaw:
    for i, objectLoc in enumerate(objectLocs):
        if (object[0] >= objectLoc[0] or object[0] <= objectLoc[3]) or \
                (object[1] >= objectLoc[1] or object[1] <= objectLoc[4]):
            objectLocs[i] = object

        else:
            objectLocs.append(object)

for object in objectLocs:
    cv2.rectangle(img, (object[0], object[1]), (object[2], object[3]), (0, 255, 0), 4)

cv2.imshow("hand", img)
cv2.imwrite("img/handDetector1.jpg", img)
cv2.waitKey()
