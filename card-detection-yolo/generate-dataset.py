import os
import cv2 as cv
import numpy as np
import imgaug as ia
import matplotlib.pyplot as PLT

DATA_DIR = "/mnt/Seagate/Code/ml-playground/card-detection-yolo/data"
TEST_DIR = "/mnt/Seagate/Code/ml-playground/card-detection-yolo/test"

CARDW=57
CARDH=87
CORNERXMIN=2
CORNERXMAX=10.5
CORNERYMIN=2.5
CORNERYMAX=23

# We convert the measures from mm to pixels: multiply by an arbitrary factor 'zoom'
ZOOM=4
CARDW*=ZOOM
CARDH*=ZOOM
CORNERXMIN=int(CORNERXMIN*ZOOM)
CORNERXMAX=int(CORNERXMAX*ZOOM)
CORNERYMIN=int(CORNERYMIN*ZOOM)
CORNERYMAX=int(CORNERYMAX*ZOOM)

REFCARD=np.array([[0, 0],[CARDW,0],[CARDW,CARDH],[0,CARDH]],dtype=np.float32)
REFCARDROT=np.array([[CARDW, 0],[CARDW,CARDH],[0,CARDH],[0,0]],dtype=np.float32)
REFCORNERHL=np.array([[CORNERXMIN, CORNERYMIN],[CORNERXMAX,CORNERYMIN],[CORNERXMAX,CORNERYMAX],[CORNERXMIN,CORNERYMAX]],dtype=np.float32)
REFCORNERLR=np.array([[CARDW-CORNERXMAX, CARDH-CORNERYMAX],[CARDW-CORNERXMIN,CARDH-CORNERYMAX],[CARDW-CORNERXMIN,CARDH-CORNERYMIN],[CARDW-CORNERXMAX,CARDH-CORNERYMIN]],dtype=np.float32)
REFCORNERS=np.array([REFCORNERHL, REFCORNERLR])

BORD_SIZE=2 # bord_size alpha=0
alphamask=np.ones((CARDH,CARDW),dtype=np.uint8)*255
cv.rectangle(alphamask,(0,0),(CARDW-1,CARDH-1),0,BORD_SIZE)
cv.line(alphamask,(BORD_SIZE*3,0),(0,BORD_SIZE*3),0,BORD_SIZE)
cv.line(alphamask,(CARDW-BORD_SIZE*3,0),(CARDW,BORD_SIZE*3),0,BORD_SIZE)
cv.line(alphamask,(0,CARDH-BORD_SIZE*3),(BORD_SIZE*3,CARDH),0,BORD_SIZE)
cv.line(alphamask,(CARDW-BORD_SIZE*3,CARDH),(CARDW,CARDH-BORD_SIZE*3),0,BORD_SIZE)

# Additional useful links
# https://stackoverflow.com/a/40204315
# https://stackoverflow.com/a/11627903

class Card:

    hullCoordinates = []
    sample = None

    def __init__(self, img):
        self.img = img
        self.copy = img.copy()

    def display(self):
        cv.imshow(self.img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    @staticmethod
    def extractPoints(event, x, y, flags, parameters):
        if event == cv.EVENT_LBUTTONDOWN:
            Card.hullCoordinates.append((x, y))
        elif event == cv.EVENT_LBUTTONUP:
            Card.hullCoordinates.append((x, y))
            cv.rectangle(Card.sample, Card.hullCoordinates[-2], Card.hullCoordinates[-1], (36,255,12), 2)

    @staticmethod
    def findConvexHull():
        while True:
            cv.imshow("Select Coordinates", Card.sample)
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                cv.destroyAllWindows()
                break
        print(Card.hullCoordinates)


def display(title, img):
    cv.imshow(title,  img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def extractImage(img):

    cannyImg = cv.Canny(cv.cvtColor(img,  cv.COLOR_BGR2GRAY), 400, 600)
    copy = img.copy()
    # https://stackoverflow.com/a/40741735
    contour = max(cv.findContours(cannyImg,  cv.RETR_LIST, cv.CHAIN_APPROX_NONE)[0], key=cv.contourArea)

    # https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html#rotation
    # rect = [(x and y of the center point),  (w and h), angle] i.e the output of minAreaRect()
    rect = cv.minAreaRect(contour)
    box = np.int0(cv.boxPoints(rect))

    rectArea = cv.contourArea(box)
    contourArea = cv.contourArea(contour)
    VALID_THRESHOLD = 0.95
    isValid = contourArea/rectArea > VALID_THRESHOLD
    if (not isValid):
        return isValid, None

    w, h = rect[1]
    if (rect[1][0] > rect[1][1]):
        h, w = rect[1]
    destPts = np.array([[0, 0], [w, 0], [w, h], [0, h]]).astype(np.float32)
    M = cv.getPerspectiveTransform(box.astype(np.float32), destPts)
    perspectiveImage = cv.warpPerspective(img, M, (int(w), int(h)))

    cv.drawContours(copy,  contour, -1, (0, 0, 255), 2, cv.LINE_AA)
    cv.drawContours(copy, [box], 0, (255, 0, 0), 2)

    display("Canny Image", cannyImg)
    display("Image with contours", copy)
    display("Perspective Transform", perspectiveImage)

    return isValid, perspectiveImage


def main():
    path = os.path.join(TEST_DIR,  "scene.png")
    img = cv.imread(path)
    isValid, perspectiveImage = extractImage(img)

    if (not isValid):
        return

    cv.namedWindow("Select Coordinates")
    cv.setMouseCallback("Select Coordinates", Card.extractPoints)

    Card.sample = perspectiveImage.copy()
    Card.findConvexHull()
    card = Card(perspectiveImage)


if __name__ == "__main__":
    main()
