import os
import cv2 as cv
import numpy as np
import imgaug as ia
import matplotlib.pyplot as plt
from collections import defaultdict
from glob import glob

DATA_DIR = "/mnt/Seagate/Code/ml-playground/card-detection-yolo/data"
TEST_DIR = "/mnt/Seagate/Code/ml-playground/card-detection-yolo/test"
DTD_DIR="/mnt/Seagate/Code/ml-playground/card-detection-yolo/dtd/images"
GENERATED_DIR = "{}/generated".format(DATA_DIR)
CARD_SUITS=['s', 'h', 'd', 'c']
CARD_VALUES=['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2']

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

    # https://stackoverflow.com/a/55153612
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


def extractImage(img, debug=False):

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

    if debug:
        display("Canny Image", cannyImg)
        display("Image with contours", copy)
        display("Perspective Transform", perspectiveImage)

    return isValid, perspectiveImage


def extractImagesFromVideo(path):

    if not os.path.exists(path):
        print('{} does not exist'.format(path))
        return

    cap = cv.VideoCapture(path)
    basename = ''.join(os.path.basename(path).split('.')[:-1])
    cards = []

    i = 0
    while (True):
        ret, frame = cap.read()
        if not ret:
            break
        isValid, processedImage = extractImage(frame)
        if not isValid:
            continue
        i += 1
        cards.append(processedImage)
        cv.imwrite('{}/{}_{}.png'.format(GENERATED_DIR, basename, i), processedImage)

    cap.release()
    return cards


def main():
    path = os.path.join(TEST_DIR,  "sample.png")
    img = cv.imread(path)
    isValid, perspectiveImage = extractImage(img, debug=True)

    if (not isValid):
        return

    # backgrounds = []
    # for dir in glob(DTD_DIR + "/*"):
    #     for file in glob(dir + "/*.jpg"):
    #         backgrounds.append(cv.imread(file))
    # print(f'Total backgrounds: {len(backgrounds)}')

    cv.namedWindow("Select Coordinates")
    cv.setMouseCallback("Select Coordinates", Card.extractPoints)
    Card.sample = perspectiveImage.copy()
    Card.findConvexHull()

    cards = defaultdict(list)
    for suit in CARD_SUITS:
        for value in CARD_VALUES:
            path = '{}/{}.avi'.format(TEST_DIR, ''.join([value, suit]))
            processedCards = extractImagesFromVideo(path)
            cards[f'{value}{suit}'].append(processedCards)



if __name__ == "__main__":
    main()
