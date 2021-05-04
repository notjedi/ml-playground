import os
import cv2 as cv
import numpy as np
from shapely.geometry import Polygon
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import matplotlib.pyplot as plt
from collections import defaultdict
from glob import glob

DATA_DIR = "/mnt/Seagate/Code/ml-playground/card-detection-yolo/data"
TEST_DIR = "/mnt/Seagate/Code/ml-playground/card-detection-yolo/test"
DTD_DIR="/mnt/Seagate/Code/ml-playground/card-detection-yolo/dtd/images"
GENERATED_DIR = "{}/generated".format(DATA_DIR)
IMG_W, IMG_H = (720, 720)
CARD_W, CARD_H = (200, 300)
OVERLAP_RATIO = 0.3
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
        processedImage = cv.resize(processedImage, (CARD_W, CARD_H))
        cards.append(processedImage)
        cv.imwrite('{}/{}_{}.png'.format(GENERATED_DIR, basename, i), processedImage)

    cap.release()
    return cards


def generate2Cards(bg, img1, img2):

    # TODO: check if the keypoints are out of the image after augmenting
    # the current values for translate_percent doesn't affect this but, if you change 
    # translate_percent to a higher value you might have to check if the keypoints are within the image
    kps = []
    start_x, start_y = ((IMG_W-CARD_W)//2, (IMG_H-CARD_H)//2)

    scaled_img1 = np.zeros((IMG_W, IMG_H, 3), dtype=np.uint8)
    scaled_img1[start_y:start_y+CARD_H, start_x:start_x+CARD_W, :] = img1
    scaled_img2 = np.zeros((IMG_W, IMG_H, 3), dtype=np.uint8)
    scaled_img2[start_y:start_y+CARD_H, start_x:start_x+CARD_W, :] = img2

    card_kps = [Keypoint(start_x, start_y), Keypoint(CARD_W+start_x, start_y), Keypoint(CARD_W+start_x, CARD_H+start_y), Keypoint(start_x, CARD_H+start_y)]
    for x, y, w, h in Card.hullCoordinates:
        kps += [Keypoint(x+start_x, y+start_y), Keypoint(x+w+start_x, y+start_y), Keypoint(x+w+start_x, y+h+start_y), Keypoint(x+start_x, y+h+start_y)]

    kps = KeypointsOnImage(kps, shape=scaled_img1.shape)
    card_kps = KeypointsOnImage(card_kps, shape=scaled_img2.shape)
    seq = iaa.Sequential([
        iaa.Affine(scale=0.7),
        iaa.Affine(rotate=(-180, 180)),
        iaa.Affine(translate_percent = {
            "x": (-0.3, 0.3),
            "y": (-0.3, 0.3)
        }),
    ])
    scale_bg =iaa.Resize({
        "height": IMG_H,
        "width": IMG_W
    })

    scaled_img1, kps1 = seq(image=scaled_img1, keypoints=kps)
    seq = seq.to_deterministic()[0]
    scaled_img2 = seq.augment_images(images=[scaled_img2])[0]
    kps2 = [seq.augment_keypoints(kps)][0]
    card_kps = [seq.augment_keypoints(card_kps)][0]
    # card_kps = seq.augment_keypoints([card_kps])[0]

    poly1 = Polygon([(k.x, k.y) for k in kps1[:4]])
    poly2 = Polygon([(k.x, k.y) for k in kps1[4:]])
    poly3 = Polygon([(k.x, k.y) for k in kps2[:4]])
    poly4 = Polygon([(k.x, k.y) for k in kps2[4:]])

    for p in [poly1, poly2]:
        for k in [poly3, poly4]:
            intersect = p.intersection(k)
            if k.area - intersect.area > k.area * OVERLAP_RATIO:
                pass
                # return None

    bg = scale_bg(image=bg)
    img_aug = np.where(scaled_img1, scaled_img1, bg)
    img_aug = np.where(scaled_img2, scaled_img2, img_aug)

    # image_after = kps1.draw_on_image(img_aug, size=5)
    # image_after = kps2.draw_on_image(img_aug, size=5)
    # image_after = card_kps.draw_on_image(image_after, size=5)
    # display("after", image_after)
    return img_aug


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
    
    for _ in range(100):
        newImg = generate2Cards(backgrounds[getRandomIndex(backgrounds)], cards['2c'][getRandomIndex(cards['2c'])], cards['2c'][getRandomIndex(cards['2c'])])
        # display("Generated image", newImg)


if __name__ == "__main__":
    main()
