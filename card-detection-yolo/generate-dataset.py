import os
import cv2 as cv
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
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
            Card.hullCoordinates.append([x, y])
        elif event == cv.EVENT_LBUTTONUP:
            cv.rectangle(Card.sample, (Card.hullCoordinates[-1][:2]), (x, y), (36,255,12), 2)
            x, y = (x - Card.hullCoordinates[-1][0], y - Card.hullCoordinates[-1][1])
            Card.hullCoordinates[-1] = tuple(Card.hullCoordinates[-1] + [x, y])

    @staticmethod
    def findConvexHull():
        while True:
            cv.imshow("Select Coordinates", Card.sample)
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                cv.destroyAllWindows()
                break


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
        # print('{} does not exist'.format(path))
        return None

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


def generate2Cards(bg, img1, img2):

    kps = []
    for x, y, w, h in Card.hullCoordinates:
        kps += [ Keypoint(x, y), Keypoint(x + w, y), Keypoint(x + w, y + h), Keypoint(x, y + h) ]

    kps_on_image = KeypointsOnImage(kps, shape=img1.shape)
    seq = iaa.Sequential([
        iaa.Affine(scale=[0.65, 1]),
        iaa.Affine(rotate=(-180, 180)),
        iaa.Affine(translate_percent = {
            "x": (-0.25, 0.25),
            "y": (-0.25, 0.25)
        }),
    ])
    seq = seq.to_deterministic()

    img1_aug, kps1_aug = seq(image=img1, keypoints=kps_on_image)
    # img2_aug, kps2_aug = seq(image=img2, keypoints=kps_on_image)
    image_before = kps_on_image.draw_on_image(img1, size=7)
    image_after = kps1_aug.draw_on_image(img1_aug, size=7)
    display("before", image_before)
    display("after", image_after)


def getRandomIndex(input):
    return np.random.randint(low=0, high=len(input)-1, size=1)[0]


def main():
    path = os.path.join(TEST_DIR,  "sample.png")
    img = cv.imread(path)
    isValid, perspectiveImage = extractImage(img, debug=True)

    if (not isValid):
        return

    backgrounds = []
    # for dir in glob(DTD_DIR + "/*"):
    # for file in glob(dir + "banded/*.jpg"):
    for file in glob(DTD_DIR + "/banded/*.jpg"):
        backgrounds.append(cv.imread(file))
    print(f'Total backgrounds: {len(backgrounds)}')

    cv.namedWindow("Select Coordinates")
    cv.setMouseCallback("Select Coordinates", Card.extractPoints)
    Card.sample = perspectiveImage.copy()
    Card.findConvexHull()

    cards = {}
    for suit in CARD_SUITS:
        for value in CARD_VALUES:
            path = '{}/{}.avi'.format(TEST_DIR, ''.join([value, suit]))
            processedCards = extractImagesFromVideo(path)
            if processedCards is not None:
                cards[f'{value}{suit}'] = processedCards
    
    newImg = generate2Cards(backgrounds[getRandomIndex(backgrounds)], cards['2c'][getRandomIndex(cards['2c'])], cards['2c'][getRandomIndex(cards['2c'])])
    display("Generated image", newImg)


if __name__ == "__main__":
    main()
