import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def get_image(imagePath):
    img = cv2.imread(imagePath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (800, 700))
    print(pytesseract.image_to_string(img))
    return img

#To detect all printed characters and handwritten english characters only
def detectChars(img):
    himg, wimg, _ = img.shape
    boxes = pytesseract.image_to_boxes(img)
    for b in boxes.splitlines():
        b = b.split(' ')
        print(b)
        x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        cv2.rectangle(img, (x, himg-y), (w, himg-h), (0, 255, 0), 1)
        cv2.putText(img, b[0], (x, himg-y+25), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)

    cv2.imshow('Detected Characters', img)
    cv2.waitKey(0)

#To detect handwritten digits
def detectDigits(img):
    himg, wimg, _ = img.shape
    #OEM runs the machine mode; 3(OEM_DEFAULT)
    #PSM : PAGE SEGMENTATION MODE; 6(PSM_SINGLE_BLOCK); 10 (single_char)
    con = r'--oem 3 --psm 6 outputbase digits'
    boxes = pytesseract.image_to_data(img, config = con)
    for x, b in enumerate(boxes.splitlines()):
        if x!=0:
            b = b.split()
            if len(b) == 12:
                print(b)
                x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                cv2.rectangle(img, (x, y), (w+x, h+y), (0, 255, 0), 2)
                cv2.putText(img, b[11], (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)

    cv2.imshow("Detected digits", img)
    cv2.waitKey(0)


img1 = get_image('CharacterRecognition\\images\\board.jpg')
detectChars(img1)
img2 = get_image('CharacterRecognition\\images\\engHand.jpg')
detectChars(img2)
img3 = get_image('CharacterRecognition\\images\\digitsHand.jpg')
detectDigits(img3)


