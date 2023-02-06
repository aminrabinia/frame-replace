import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def mask_to_image(img, mask):
    # convert mask to black and white image, easier to find corners
    alpha = 0.99
    for c in range(3):
        img[:, :, c] = np.where(mask == 1,
                                img[:, :, c] * (1-alpha) + alpha * 255,
                                img[:, :, c] * (1-alpha) + alpha * 1)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((5, 5), np.float32) / 15
    filtered = cv2.filter2D(gray, -1, kernel)
    ret, thresh = cv2.threshold(filtered, 250, 255, cv2.THRESH_OTSU)
    return thresh

# def vis_corners(canvas, approx_corners):
    # print('\nThe updated corner points are ...\n')
    for index, c in enumerate(approx_corners):
        character = chr(65 + index)
        # print(character, ':', c)
        cv2.putText(canvas, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, '.', tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
    plt.figure(figsize = (15,15))
    plt.imshow(canvas)
    plt.show()
    
def detect_corners(canvas, img):
    # find four corners of the door
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)[-1]
    cv2.drawContours(canvas, cnt, -1, (255, 255, 0), 2)

    epsilon = 0.0001 * cv2.arcLength(cnt, True)
    approx_corners = cv2.approxPolyDP(cnt, epsilon, True)
    approx_corners = sorted(np.concatenate(approx_corners).tolist())
    # sorting corners to 4 points [from upper left clockwise]
    rect = np.zeros((4, 2), np.int32)
    pts = np.array(approx_corners)
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    approx_corners = rect.tolist()
    # vis_corners(canvas, approx_corners)
    return approx_corners


def skew_img(img, bg_img, corners):
    # skew door image to fit in the bigger image
    rows, cols, ch = img.shape 
    pts1 = np.float32(
        [[0,      0],
        [cols,    0],
        [cols, rows],
        [0,    rows]])    
    # adjust corners to start from (min_x,min_y) in doorimg
    minvar = np.argmin(corners, axis=0) # [min_x, min_y]
    min_x = corners[minvar[0]][0]
    min_y = corners[minvar[1]][1]
    new_corners = []
    offset = (corners[1][0]-corners[0][0]) * 0.0001 # set a ratio of width as offset
    for point in corners:
      new_corners.append([point[0]- (min_x + offset), 
                        point[1]- (min_y + offset)])
    scale_w = 0.0001
    scale_h = 0.0001
    new_corners[1][0] += (new_corners[1][0] - new_corners[0][0]) * scale_w # increase top width (x of B)
    new_corners[2][0] += (new_corners[2][0] - new_corners[3][0]) * scale_w # increase down width (x of C)
    new_corners[2][1] += (new_corners[2][1] - new_corners[1][1]) * scale_h # increase right height (y of C)
    new_corners[3][1] += (new_corners[3][1] - new_corners[0][1]) * scale_h # increase left height (y of D)
    
    pts2 = np.float32(new_corners)
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img, M, (cols, rows), bg_img, borderMode=cv2.BORDER_TRANSPARENT)

    return dst

# def change_brightness(img, img2, alpha = 1):
    # adjust the img2 birghness to bigger image light
    brt = np.average(norm(img, axis=2)) / np.sqrt(3)
    brt = brt - 150
    img2 = cv2.addWeighted(img2, alpha, img2,0, brt)
    return img2

def apply_mask_on_image(image, img2, box, corners, alpha):

    x1, y1, x2, y2 = box
    height, width = (y2-y1), (x2-x1)

    img2 = cv2.resize(img2, (width, height), interpolation = cv2.INTER_AREA)
    background_image = image[y1:y2, x1:x2] # skew_img needs to have behind the img2
    img_pic = skew_img(img2, background_image, corners)

    for c in range(3):
        image[y1:y2, x1:x2, c] = alpha * img_pic[:, :, c] 
                                 
    return image

def display_instances(image1, img2, boxes, masks):

    i = 0
    mask = masks[i, :, :]
    box = boxes[i]

    im = image1.copy()
    canvas = image1.copy()
    image_mask = mask_to_image(im, mask)
    corners = detect_corners(canvas, image_mask)

    image = apply_mask_on_image(image1, img2, box, corners, alpha=0.99)

    return image