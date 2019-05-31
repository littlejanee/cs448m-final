import cv2
from skimage import measure
import numpy as np
from scipy import stats

fgbg = cv2.createBackgroundSubtractorMOG2()

def find_marker(frame):
    fgmask = fgbg.apply(frame)
    masked = cv2.bitwise_and(frame, frame, mask=fgmask)

    hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, sat_bright = cv2.threshold(sat, 200, 255, cv2.THRESH_TOZERO)
    sat_bright = cv2.erode(sat_bright, None, iterations=2)
    sat_bright = cv2.dilate(sat_bright, None, iterations=2)

    cnts, _ = cv2.findContours(sat_bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_srt = sorted([(c, cv2.contourArea(c)) for c in cnts], key=lambda t: t[1])
    if len(cnts) > 0:
        (c, area) = cnts_srt[-1]

        if area > 1000:
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy

    return None

class History:
    def __init__(self, n=10):
        self.pts = []
        self.n = n

    def shift(self, pt):
        self.pts.append(pt)
        if len(self.pts) > self.n:
            del self.pts[0]

    def last(self):
        return self.pts[-1] if len(self.pts) > 0 else None

def main():
    cam = cv2.VideoCapture(1)
    drawn = np.zeros((1080, 1920))
    history = History()

    while True:
        result, frame = cam.read()
        if not result:
            raise Exception("Cam read failed")

        point = find_marker(frame)
        if point is not None:
            if history.last() is not None:
                cv2.line(drawn, history.last(), point, (255, 255, 255), 3)
            history.shift(point)

        cv2.imshow('frame', drawn)
        cv2.waitKey(30)

if __name__ == "__main__":
    main()
