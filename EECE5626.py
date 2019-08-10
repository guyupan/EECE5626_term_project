#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libs
import numpy as np
import os
import cv2
import utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[2]:


#camera calibration
cal_images = utils.get_images_by_dir('camera_cal')


# In[3]:


cal_images2 = []
for img in cal_images:
    img_2 = img[:,:,[2,1,0]]
    cal_images2.append(img_2)


# In[4]:


object_points,img_points = utils.calibrate(cal_images,grid=(9,6))


# In[5]:


undistorted_cal_images = []
for img in cal_images:
    image = utils.cal_undistort(img, object_points, img_points)
    image = image[:,:,[2,1,0]]
    undistorted_cal_images.append(image)


# In[6]:


plt.figure(figsize=(20,68))
for i in range(1,2):
    plt.subplot(len(cal_images),2,2*i+1)
    plt.title('before undistorted')
    plt.imshow(cal_images2[i])
    plt.subplot(len(cal_images),2,2*i+2)
    plt.title('after undistorted')
    plt.imshow(undistorted_cal_images[i])
plt.savefig('camera_calibration.png')


# In[7]:


#undistorted test images
test_imgs = utils.get_images_by_dir('test_images')
test_imgs2 = []
for img in test_imgs:
    img_2 = img[:,:,[2,1,0]]
    test_imgs2.append(img_2)


# In[8]:


undistorted_test_images = []
for img in test_imgs:
    image = utils.cal_undistort(img, object_points, img_points)
    image = image[:,:,[2,1,0]]
    undistorted_test_images.append(image)


# In[9]:


plt.figure(figsize=(20,40))
for i in range(0,(len(undistorted_test_images)-1)):
    plt.subplot(len(undistorted_test_images),2,2*i+1)
    plt.title('original image')
    plt.imshow(test_imgs2[i])
    plt.subplot(len(undistorted_test_images),2,2*i+2)
    plt.title('after undistortion')
    plt.imshow(undistorted_test_images[i])
plt.savefig('test_image_undistorted.png')


# In[10]:


#x_thresholding methdods
x_thresh_method = []
for img in test_imgs2:
    x_thresh = utils.abs_sobel_thresh(img, orient='x', thresh_min=35, thresh_max=100)
    x_thresh_method.append(x_thresh)


# In[11]:


plt.figure(figsize=(20,40))
for i in range(0,(len(undistorted_test_images))):
    plt.subplot(len(undistorted_test_images),2,2*i+1)
    plt.title('original image')
    plt.imshow(test_imgs2[i])
    plt.subplot(len(undistorted_test_images),2,2*i+2)
    plt.title('after x_thresh_method')
    plt.imshow(x_thresh_method[i], cmap = 'gray')
plt.savefig('x_thresh_method.png')


# In[12]:


#mag_thresholding methdods
mag_thresh_method = []
for img in test_imgs2:
    mag_thresh = utils.mag_thresh(img, sobel_kernel=9, mag_thresh=(50, 100))
    mag_thresh_method.append(mag_thresh)


# In[13]:


plt.figure(figsize=(20,40))
for i in range(0,(len(undistorted_test_images))):
    plt.subplot(len(undistorted_test_images),2,2*i+1)
    plt.title('original image')
    plt.imshow(test_imgs2[i])
    plt.subplot(len(undistorted_test_images),2,2*i+2)
    plt.title('after mag_thresh_method')
    plt.imshow(mag_thresh_method[i], cmap = 'gray')
plt.savefig('mag_thresh_method.png')


# In[14]:


#hls_thresholding methdods
hls_thresh_method = []
for img in test_imgs2:
    hls_thresh = utils.hls_select(img,channel='s',thresh=(180, 255))
    hls_thresh_method.append(hls_thresh)

plt.figure(figsize=(20,40))
for i in range(0,(len(undistorted_test_images))):
    plt.subplot(len(undistorted_test_images),2,2*i+1)
    plt.title('original image')
    plt.imshow(test_imgs2[i])
    plt.subplot(len(undistorted_test_images),2,2*i+2)
    plt.title('after hls_thresh_method')
    plt.imshow(hls_thresh_method[i], cmap = 'gray')
plt.savefig('hls_thresh_method.png')


# In[15]:


#dir_thresholding methdods
dir_thresh_method = []
for img in test_imgs2:
    dir_thresh = utils.dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
    dir_thresh_method.append(dir_thresh)

plt.figure(figsize=(20,40))
for i in range(0,(len(undistorted_test_images))):
    plt.subplot(len(undistorted_test_images),2,2*i+1)
    plt.title('original image')
    plt.imshow(test_imgs2[i])
    plt.subplot(len(undistorted_test_images),2,2*i+2)
    plt.title('after dir_thresh_method')
    plt.imshow(dir_thresh_method[i], cmap = 'gray')
plt.savefig('dir_thresh_method.png')


# In[16]:


#lab_thresholding methdods
lab_thresh_method = []
for img in test_imgs2:
    lab_thresh = utils.lab_select(img, thresh=(155, 200))
    lab_thresh_method.append(lab_thresh)

plt.figure(figsize=(20,40))
for i in range(0,(len(undistorted_test_images))):
    plt.subplot(len(undistorted_test_images),2,2*i+1)
    plt.title('original image')
    plt.imshow(test_imgs2[i])
    plt.subplot(len(undistorted_test_images),2,2*i+2)
    plt.title('after lab_thresh_method')
    plt.imshow(lab_thresh_method[i], cmap = 'gray')
plt.savefig('lab_thresh_method.png')


# In[17]:


#luv_thresholding methdods
luv_thresh_method = []
for img in test_imgs2:
    luv_thresh = utils.luv_select(img, thresh=(225, 255))
    luv_thresh_method.append(luv_thresh)

plt.figure(figsize=(20,40))
for i in range(0,(len(undistorted_test_images))):
    plt.subplot(len(undistorted_test_images),2,2*i+1)
    plt.title('original image')
    plt.imshow(test_imgs2[i])
    plt.subplot(len(undistorted_test_images),2,2*i+2)
    plt.title('after luv_thresh_method')
    plt.imshow(luv_thresh_method[i], cmap = 'gray')
plt.savefig('luv_thresh_method.png')


# In[18]:


def thresholding(img):
    x_thresh = utils.abs_sobel_thresh(img, orient='x', thresh_min=10 ,thresh_max=230)
    mag_thresh = utils.mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 150))
    dir_thresh = utils.dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
    hls_thresh = utils.hls_select(img, thresh=(180, 255))
    lab_thresh = utils.lab_select(img, thresh=(155, 200))
    luv_thresh = utils.luv_select(img, thresh=(225, 255))
    #Thresholding combination
    threshholded = np.zeros_like(x_thresh)
    threshholded[((x_thresh == 1) & (mag_thresh == 1)) | ((dir_thresh == 1) & (hls_thresh == 1)) | (lab_thresh == 1) | (luv_thresh == 1)] = 1

    return threshholded

#combined thresholding method
combined_thresh_method = []
for img in test_imgs2:
    combined_thresh = thresholding(img)
    combined_thresh_method.append(combined_thresh)

plt.figure(figsize=(20,40))
for i in range(0,(len(undistorted_test_images))):
    plt.subplot(len(undistorted_test_images),2,2*i+1)
    plt.title('original image')
    plt.imshow(test_imgs2[i])
    plt.subplot(len(undistorted_test_images),2,2*i+2)
    plt.title('after combined_thresh_method')
    plt.imshow(combined_thresh_method[i], cmap = 'gray')
plt.savefig('combined_thresh_method.png')


# In[19]:


#use perspective transform to achieve bird's eye images
M,Minv = utils.get_M_Minv()


# In[20]:


perspective_transformed_images = []
for img in combined_thresh_method:
    perspective_transformed = cv2.warpPerspective(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    perspective_transformed_images.append(perspective_transformed)

plt.figure(figsize=(20,40))
for i in range(0,(len(undistorted_test_images))):
    plt.subplot(len(undistorted_test_images),2,2*i+1)
    plt.title('combined_thresh_image')
    plt.imshow(combined_thresh_method[i], cmap = 'gray')
    plt.subplot(len(undistorted_test_images),2,2*i+2)
    plt.title('after_perspective_transform')
    plt.imshow(perspective_transformed_images[i], cmap = 'gray')
plt.savefig('perspective_transformed_images.png')


# In[21]:


hist = []
for img in perspective_transformed_images:
    image = np.sum(img[img.shape[0]//2:,:], axis=0)
    hist.append(image)
    
plt.figure(figsize=(20,40))
for i in range(0,(len(undistorted_test_images))):
    plt.subplot(len(undistorted_test_images),2,2*i+1)
    plt.title('perspected_transformed_binary_image')
    plt.imshow(perspective_transformed_images[i], cmap = 'gray')
    plt.subplot(len(undistorted_test_images),2,2*i+2)
    plt.title('histogram')
    plt.plot(hist[i])
plt.savefig('histogram.png')


# In[22]:


#lane fitting
plt.figure(figsize=(20,40))
i=0
for binary_warped in perspective_transformed_images:
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    plt.subplot(len(perspective_transformed_images),2,2*i+1)
    plt.title('perspective_transformed_binary_image')
    plt.imshow(binary_warped,cmap ='gray')
    
    plt.subplot(len(perspective_transformed_images),2,2*i+2)
    i+=1
    plt.title('lane fitting')
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
plt.savefig('lane fitting.png')


# In[23]:


for binary_warped in perspective_transformed_images:
    left_fit, right_fit, left_lane_inds, right_lane_inds = utils.find_line(binary_warped)
    curvature,distance_from_center = utils.calculate_curv_and_pos(binary_warped,left_fit, right_fit)


# In[24]:


plt.figure(figsize=(20,40))
for i in range(0,(len(undistorted_test_images))):
    left_fit, right_fit, left_lane_inds, right_lane_inds = utils.find_line(perspective_transformed_images[i])
    curvature,distance_from_center = utils.calculate_curv_and_pos(perspective_transformed_images[i],left_fit, right_fit)
    result = utils.draw_area(undistorted_test_images[i],perspective_transformed_images[i],Minv,left_fit, right_fit)
    img = utils.draw_values(result, curvature, distance_from_center)
    
    plt.subplot(len(undistorted_test_images),1,i+1)
    plt.title('result')
    plt.imshow(img)
    
plt.savefig('result.png')    


# In[ ]:




