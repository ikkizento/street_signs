import package as pack
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

images = [cv2.imread(file) for file in glob.glob("test/*.jpg")]
templates = [cv2.imread(file) for file in glob.glob("package/templates/*.png")]

#setting the white color
lower_white = np.array([13,13,0])
upper_white = np.array([39,20,250])

#choosing images
img = images[8]

#converting image and template to be comparable
dilation = pack.editing_image(img,lower_white,upper_white)

#finding contours in dilation image
edges = cv2.Canny(img,400,500)
image, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

new_contours = pack.make_contours(img,contours)

#find array of possible street signs and index of their contours
cropped_images = pack.cropp_squares(img,new_contours) 


idx = 0
min_diff_index = 0
min_diff = 100000
template_index = 0
for i in range(len(templates)):
	tem = pack.editing_image(templates[i],lower_white,upper_white)
	#find height and width of template, so we can compare them with what we find on test image
	diff_array = pack.find_diff_array(cropped_images,tem,lower_white,upper_white) 
	#find the one which is most possible
	new_min_diff, new_min_diff_index = pack.find_min_dif_index(diff_array)
	if (new_min_diff < min_diff):
		min_diff_index = new_min_diff_index
		min_diff = new_min_diff
		template_index = i
	idx +=1

result = cropped_images[min_diff_index]
cv2.imwrite('result.png', result)

#pack.draw_all_contours(img,new_contours)
cv2.drawContours(img,[new_contours[min_diff_index]], 0, (0,255,0), 3)


fig, axes = plt.subplots(1, 1, figsize=(10, 6))
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(result,cmap = 'gray')
plt.title('Street signs'), plt.xticks([]), plt.yticks([])
plt.show()