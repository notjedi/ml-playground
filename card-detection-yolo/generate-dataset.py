import os
import random
import cv2 as cv
import numpy as np
import imgaug.augmenters as iaa
from glob import glob
from tqdm import tqdm, trange
from collections import defaultdict
from shapely.geometry import Polygon
from pascal_voc_writer import Writer
from imgaug.augmentables import Keypoint, KeypointsOnImage
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

DATA_DIR = "/mnt/Seagate/Code/ml-playground/card-detection-yolo/data"
TEST_DIR = "/mnt/Seagate/Code/ml-playground/card-detection-yolo/test"
DTD_DIR="/mnt/Seagate/Code/ml-playground/card-detection-yolo/dtd/images"
GENERATED_DIR = "{}/generated".format(DATA_DIR)

IMG_W, IMG_H = (720, 720)
CARD_W, CARD_H = (200, 300)
START_X, START_Y = ((IMG_W-CARD_W)//2, (IMG_H-CARD_H)//2)
OVERLAP_RATIO = 0.2

CARD_SUITS=['s', 'h', 'd', 'c']
CARD_VALUES=['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2']

# Additional useful links
# https://stackoverflow.com/a/40204315
# https://stackoverflow.com/a/11627903

class Card:

    card_kps = [Keypoint(START_X, START_Y), Keypoint(CARD_W+START_X, START_Y), Keypoint(CARD_W+START_X, CARD_H+START_Y), Keypoint(START_X, CARD_H+START_Y)]
    hullCoordinates = []
    sample = None
    kps = []

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

    @staticmethod
    def calcKps():
        for x, y, w, h in Card.hullCoordinates:
            Card.kps += [Keypoint(x+START_X, y+START_Y), Keypoint(x+w+START_X, y+START_Y), Keypoint(x+w+START_X, y+h+START_Y), Keypoint(x+START_X, y+h+START_Y)]


def display(title, img):
    cv.imshow(title,  img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def getRandomIndex(input):
    return np.random.randint(0, len(input)-1, size=1)[0]

def getRandomCard(cards):
    name = random.choice(list(cards.keys()))
    return (cards[name][getRandomIndex(cards[name])],  name)

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
    # basename = ''.join(os.path.basename(path).split('.')[:-1])
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
        # cv.imwrite('{}/{}_{}.png'.format(GENERATED_DIR, basename, i), processedImage)

    cap.release()
    return cards

def kpsToBB(name, kps):
    kpx = [kp.x for kp in kps]
    kpy = [kp.y for kp in kps]
    minx, miny = min(kpx), min(kpy)
    maxx, maxy = max(kpx), max(kpy)
    return BoundingBox(label=name, x1=minx, y1=miny, x2=maxx, y2=maxy)

def createVocFile(fileName, listBbs):
    vocFile = Writer(fileName, IMG_W, IMG_H)
    for bbs in listBbs:
        vocFile.addObject(bbs.label, bbs.x1, bbs.y1, bbs.x2, bbs.y2)
    vocFile.save(fileName.replace('png', 'xml'))


def generate2Cards(bg, img1, img2, name1, name2, debug=False):

    # TODO: check if the keypoints are out of the image after augmenting
    # the current values for translate_percent doesn't affect this but, if you change 
    # translate_percent to a higher value you might have to check if the keypoints are within the image
    seq = iaa.Sequential([
        # TODO: add Flipud() maybe
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

    scaled_img1 = np.zeros((IMG_W, IMG_H, 3), dtype=np.uint8)
    scaled_img2 = np.zeros((IMG_W, IMG_H, 3), dtype=np.uint8)
    scaled_img1[START_Y:START_Y+CARD_H, START_X:START_X+CARD_W, :] = img1
    scaled_img2[START_Y:START_Y+CARD_H, START_X:START_X+CARD_W, :] = img2


    kps = KeypointsOnImage(Card.kps, shape=scaled_img1.shape)
    card_kps = KeypointsOnImage(Card.card_kps, shape=scaled_img2.shape)

    scaled_img1, kps1 = seq(image=scaled_img1, keypoints=kps)
    # https://github.com/aleju/imgaug/issues/112#issuecomment-376561789
    seq = seq.to_deterministic()
    scaled_img2 = seq.augment_image(scaled_img2)
    kps2 = seq.augment_keypoints([kps])[0]
    card_kps = seq.augment_keypoints([card_kps])[0]

    img2Poly = Polygon([(k.x, k.y) for k in card_kps])
    selected = [kps2[:4], kps2[4:]]

    for i in range(0, 8, 4):
        pts = kps1[i:i+4]
        poly = Polygon([(p.x, p.y) for p in pts])
        intersect = poly.intersection(img2Poly)
        if intersect.area < poly.area * OVERLAP_RATIO:
            selected.append(pts)
    
    bg = scale_bg(image=bg)
    img_aug = np.where(scaled_img1, scaled_img1, bg)
    img_aug = np.where(scaled_img2, scaled_img2, img_aug)
    bbs = [kpsToBB(name2, kp) for kp in selected[:2]]
    bbs += [kpsToBB(name1, kp) for kp in selected[2:]]

    if debug:
        bbs_img = BoundingBoxesOnImage(bbs, shape=img_aug.shape)
        image_after = kps2.draw_on_image(img_aug, size=5)
        image_after = card_kps.draw_on_image(image_after, size=5)
        image_after = bbs_img.draw_on_image(img_aug)
        display("after", image_after)

    return (img_aug, bbs)


def main():
    testFilePath = os.path.join(TEST_DIR,  "sample.png")
    testImg = cv.imread(testFilePath)
    _, perspectiveImage = extractImage(testImg)

    backgrounds = []
    # for dir in glob(DTD_DIR + "/*"):
    # for file in glob(dir + "banded/*.jpg"):
    # TODO: change to whole directory instead of only `banded/*`
    for file in tqdm(glob(DTD_DIR + "/banded/*.jpg")):
        backgrounds.append(cv.imread(file))
    print(f'Total backgrounds: {len(backgrounds)}')

    cv.namedWindow("Select Coordinates")
    cv.setMouseCallback("Select Coordinates", Card.extractPoints)
    Card.sample = perspectiveImage.copy()
    Card.findConvexHull()
    Card.calcKps()

    cards = defaultdict(list)
    for suit in CARD_SUITS:
        for value in CARD_VALUES:
            val = f'{value}{suit}'
            path = '{}/{}.avi'.format(TEST_DIR, val)
            processedCards = extractImagesFromVideo(path)
            if processedCards == None:
                continue
            cards[val] += processedCards

    for i in trange(1, 101):
        img1, name1 = getRandomCard(cards)
        img2, name2 = getRandomCard(cards)
        img, bbs = generate2Cards(backgrounds[getRandomIndex(backgrounds)], img1, img2, name1, name2)
        imgPath = '{}/{}.png'.format(GENERATED_DIR, i)
        cv.imwrite(imgPath, img)
        createVocFile(imgPath, bbs)
        # display("Generated image", img)


if __name__ == "__main__":
    main()
