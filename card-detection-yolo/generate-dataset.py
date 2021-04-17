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

def display(img):
    cv.imshow("img",  img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def main():
    path = os.path.join(TEST_DIR,  "scene.png")
    img = cv.imread(path)
    copy = img.copy()
    cannyImg = cv.Canny(cv.cvtColor(img,  cv.COLOR_BGR2GRAY), 400, 600)
    display(cannyImg)

    # https://stackoverflow.com/a/40741735
    contour = max(cv.findContours(cannyImg,  cv.RETR_LIST, cv.CHAIN_APPROX_NONE)[0], key=cv.contourArea)
    cv.drawContours(copy,  contour, -1, (0, 0, 255), 2, cv.LINE_AA)
    contourArea = cv.contourArea(contour)

    # https://stackoverflow.com/a/40204315
    # x, y,w,h = cv.boundingRect(contour)
    # cv.rectangle(img,  (x,y), (x+w,y+h), (255, 0, 0), 2)
    # rectArea = w * h
    # display(img)

    # using this method instead of the above one cause it can find rotated rectangles and also gives the min area
    # https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
    # rect = [(x and y of the center point),  (w and h), angle] i.e the output of minAreaRect()
    rect = cv.minAreaRect(contour)
    box = np.int0(cv.boxPoints(rect))
    cv.drawContours(copy, [box], 0, (255, 0, 0), 2)
    rectArea = cv.contourArea(box)
    # rectArea = rect[1][0] * rect[1][1]

    VALID_THRESHOLD = 0.95
    isValid = contourArea/rectArea > VALID_THRESHOLD
    display(copy)

    if (isValid):
        (w, h) = rect[1]
        if (w > h):
            M = cv.getPerspectiveTransform(np.float32(box), REFCARD)
        else:
            M = cv.getPerspectiveTransform(np.float32(box), REFCARDROT)
        imgwarp = cv.warpPerspective(img, M, (CARDW, CARDH))
        imgwarp = cv.cvtColor(imgwarp, cv.COLOR_BGR2BGRA)
        display(imgwarp)
        
        cnta = contour.reshape(1, -1, 2).astype(np.float32)
        cntwarp = cv.perspectiveTransform(cnta, M)
        cntwarp = cntwarp.astype(np.int16)

        alphachannel = np.zeros(imgwarp.shape[:2], dtype=np.uint8)
        # cv.drawContours(alphachannel, [cntwarp], 0, (255, 0, 0), -1)
        
        alphachannel = cv.bitwise_and(alphachannel, alphamask)
        
        imgwarp[:,:,3] = alphachannel
        display(imgwarp)
    else:
        cv.destroyAllWindows()
        return

    img = cropImage(img, rect)
    display(img)

    cv.destroyAllWindows()

def cropImage(img, rect):

    # https://stackoverflow.com/a/11627903
    (center), (width, height), theta = rect
    width, height = int(width), int(height)

    matrix = cv.getRotationMatrix2D(center=center, angle=theta, scale=1)
    img = cv.warpAffine(src=img, M=matrix, dsize=np.flip(img.shape[:2])) # expects shape in (length, height)

    x = int(center[0] - width/2)
    y = int(center[1] - height/2)
    img = img[y:y+height, x:x+width]
    # rotate image if it's horizontal, i.e if it's w > h
    if (img.shape[0] < img.shape[1]):
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    return img


if __name__ == "__main__":
    main()
