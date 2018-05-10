import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_contours(img,contours):
	new_contours = []
	for cnt in contours:
		new_cnt = [[0 for x in range(2)] for y in range(4)]
		min_height = np.size(img, 0)
		min_width = np.size(img, 1)
		max_height = 0
		max_width = 0
		for i in range(len(cnt)):

			if (cnt[i][0][0] > max_width):
				max_width = cnt[i][0][0]

			if (cnt[i][0][0] < min_width):
				min_width = cnt[i][0][0]

			if (cnt[i][0][1] > max_height):
				max_height = cnt[i][0][1]

			if (cnt[i][0][1] < min_height):
				min_height = cnt[i][0][1]

		new_cnt[0] = [[min_width,min_height]]
		new_cnt[2] = [[max_width,max_height]]
		new_cnt[1] = [[min_width,max_height]]
		new_cnt[3] = [[max_width,min_height]]
		new_cnt = np.array(new_cnt)
		#cv2.drawContours(img,[new_cnt], 0, (0,255,0), 3)

		w = max_width - min_width
		h = max_height - min_height
		if w>25 and h>25:
			new_contours.append(new_cnt) 
	return new_contours

def cropp_squares(img,contours):
	idx = 0
	cropped_images = []
	contours_idx = []
	for i in range(len(contours)):
		x,y,w,h = cv2.boundingRect(contours[i])
		if w>0 and h>0:
			idx+=1
			new_img=img[y:y+h,x:x+w]

			# comparing_new_img = new_img
			# comparing_new_img = editing_image(comparing_new_img)
			# loop = check_borders(comparing_new_img)
			# new_img = update_borders(new_img,loop)
			cropped_images.append(new_img)
			#cv2.imwrite(str(idx) + '_cropped.png', new_img)
	return cropped_images

def editing_image(img,lower,upper):
	blur = cv2.GaussianBlur(img,(1,1),20)
	HSV_img = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

	mask = cv2.inRange(HSV_img, lower, upper)

	kernel= np.ones((3,3),np.uint8)
	kernel2= np.ones((4,4),np.uint8)
	erosion = cv2.erode(mask,kernel,iterations = 1)
	dilation = cv2.dilate(erosion,kernel2,iterations = 6)

	return dilation

def check_borders(img):
	count_black = 0
	all_white = 0
	loop = 0
	while (count_black >= all_white/2):
		current_i = len(img[0])-loop
		current_j = len(img[1])-loop
		all_white = (current_j+current_j)*2-4
		for i in range(current_i - 1):
			if (img[i][0] == 255):
				count_black += 1
			if (img[i][current_j - 1] == 0):
				count_black += 1
		for j in range(current_j - 1):
			if (img[0][j] == 255):
				count_black += 1
			if (img[current_i - 1][j] == 0):
				count_black += 1
		loop += 1
	return loop

def find_diff_array(cropped_images,template,lower,upper):
	diff_array = []
	for i in range(len(cropped_images)):
		comparing_image = cropped_images[i]
		comparing_image = resize_image(comparing_image,template)
		comparing_image = editing_image(comparing_image,lower,upper)
		diff = cv2.subtract(comparing_image,template)
		#cv2.imshow(str(i)+'diff',diff)
		#cv2.imshow(str(i) + '_diffed',diff)
		pixel_diff = 0
		for diff_i in diff:
			for dif_j in diff_i:
				pixel_diff += dif_j
		diff_array.append(pixel_diff)
	return diff_array

def find_min_dif_index(diff_array):
	min_diff = 100000
	min_diff_index = 0
	for i in range(len(diff_array)-1):
		if (diff_array[i] < min_diff):
			min_diff = diff_array[i]
			min_diff_index = i
	return min_diff, min_diff_index

def update_borders(img,loop):
	img_without_borders = img[0+loop:len(img[0])-loop,0+loop:len(img[1])-loop]
	return img_without_borders

def resize_image(img,tem):
	height = np.size(tem, 0)
	width = np.size(tem, 1)
	new_img = img[0+5:len(img[0])-5,0+5:len(img[1])-5]
	new_img = cv2.resize(new_img, (width, height)) 
	return new_img

def draw_all_contours(img,contours):
	for cnt in contours:
		cv2.drawContours(img,[cnt], 0, (0,255,0), 3)

def group_rectangles(contours):
	size = len(contours)
	rects = []
	for i in range(size):
		rects.append(contours[i])
	new_rect = cv2.groupRectangles(rects, 10, 0.2)
	return new_rect