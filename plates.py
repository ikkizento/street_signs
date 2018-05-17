import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import cv2
import json


class VisualHelper:
    def draw_ctrs(self, img, ctrs):
        # rgb4presentation = cv2.merge((img, img, img))
        # to_show = cv2.drawContours
        # (rgb4presentation, ctrs, -1, (255, 0, 0), 3)
        to_show = cv2.drawContours(img, ctrs, -1, (255, 0, 0), 3)
        return to_show

    def show_arr(self, to_show):
        for i in range(len(to_show)):
            cv2.imshow(str(i), to_show[i])


class RectHelper:
    def iou(self, r1, r2):
        x1, y1, w1, h1 = r1
        x2, y2, w2, h2 = r2
        and_x1, and_y1 = max(x1, x2), max(y1, y2)
        and_x2, and_y2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        and_w = and_x2 - and_x1
        and_h = and_y2 - and_y1
        if and_w <= 0 or and_h <= 0:
            return 0
        and_area = and_w * and_h
        area1 = w1 * h1
        area2 = w2 * h2
        or_area = area1 + area2 - and_area

        return and_area / or_area

    def is_ctrs_similar(self, a, b):
        brect_a = cv2.boundingRect(a)
        brect_b = cv2.boundingRect(b)

        return self.iou(brect_a, brect_b) > 0.75


class PlateFounder:

    def __init__(self):
        self.vh = VisualHelper()
        self.rh = RectHelper()

    def find_plates(self, img):
        blurred = cv2.medianBlur(img, 5)
        filtered = self.binarize_colored(blurred)
        return self.find_squares(filtered)

    def binarize_colored(self, img, morph_in=7, morph_out=7):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 0])
        upper = np.array([179, 50, 255])

        mask = cv2.inRange(hsv, lower, upper)

        kernel_in = np.ones((morph_in, morph_in), np.uint8)
        kernel_out = np.ones((morph_out, morph_out), np.uint8)
        # cv2.imshow('img0',mask)
        erased = cv2.erode(mask, kernel_in, iterations=1)
        # cv2.imshow('img',erased)
        dilated = cv2.dilate(erased, kernel_out, iterations=1)
        # cv2.imshow('img2',dilated)

        res = cv2.bitwise_and(img, img, mask=dilated)
        return res

    def filter_ctrs(self, ctrs):
        dst = []
        for item in ctrs:
            good = True
            for already in dst:
                if self.rh.is_ctrs_similar(item, already):
                    good = False
                    break
            if good:
                dst.append(item)
        return dst

    def angle_cos(self, p0, p1, p2):
        d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
        return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1)*np.dot(d2, d2)))

    def find_squares(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        squares = []
        w, h = img.shape
        for thrs in range(0, 255, 26):
            if thrs == 0:
                continue
            else:
                retval, bin = cv2.threshold(img, thrs, 255, cv2.THRESH_BINARY)
            bin, contours, hierarchy = cv2.findContours(
                bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1200 \
                        and cv2.contourArea(cnt) < w * h / 10 \
                        and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max(
                        [self.angle_cos(cnt[i],
                                        cnt[(i + 1) % 4],
                                        cnt[(i + 2) % 4])
                         for i in range(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)

        return self.filter_ctrs(squares)
        # return squares


def sample(fpath):
    img = cv2.imread(fpath)
    engine = PlateFounder()
    visualizer = VisualHelper()
    rects = engine.find_plates(img)
    with_rects = visualizer.draw_ctrs(img, rects)
    visualizer.show_arr([with_rects])
    cv2.waitKey(0)


def benchmark(data_paths, markup_path, show_comparasion=False):
    fp = 0
    tp = 0
    fn = 0
    tn = 0

    testdata = [cv2.imread(path) for path in data_paths]
    engine = PlateFounder()
    rh = RectHelper()
    vh = VisualHelper()
    answers = [engine.find_plates(img) for img in testdata]

    with open('markup.json') as f:
        json_data = json.load(f)

    if show_comparasion:
        for img, path in zip(testdata, data_paths):
            markup_rects = [(i['x'], i['y'], i['w'], i['h'])
                            for i in json_data[path[5:]]]
            found_rects = engine.find_plates(img)
            for rect in found_rects:
                x, y, w, h = cv2.boundingRect(rect)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for rect in markup_rects:
                x, y, w, h = rect
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        vh.show_arr(testdata)

    for path, f_ctrs in zip(data_paths, answers):
        markup_rects = [(i['x'], i['y'], i['w'], i['h'])
                        for i in json_data[path[5:]]]
        for m_rect in markup_rects:
            found = False
            for ctr in f_ctrs:
                if rh.iou(cv2.boundingRect(ctr), m_rect) > 0.75:
                    found = True
            if found:
                tp += 1
                print(path[5:], 'tp')
            else:
                fn += 1
                print(path[5:], 'fn')

        for ctr in f_ctrs:
            found = False
            for m_rect in markup_rects:
                if rh.iou(cv2.boundingRect(ctr), m_rect) > 0.75:
                    found = True
            if not found:
                fp += 1
                print(path[5:], 'fp')

    precision = tp / (tp + fp)
    recall = tp / (fn + tp)

    return precision, recall


if __name__ == '__main__':
    # sample('test/0.jpg')

    testdata = ['test/' + str(i) + '.jpg' for i in range(1)]
    markup_path = 'markup.json'
    precision, recall = benchmark(testdata, markup_path, show_comparasion=True)
    print('precision: ', precision, 'recall: ', recall)
    cv2.waitKey(0)
