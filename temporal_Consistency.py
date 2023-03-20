import os 
import cv2 as cv
import numpy as np
import scipy.misc

w_point = 600 # Choose weight-coordinate to be transferred to profile
img_height = 720

# #----------------------------------------------------------------------------
# img_root = '/media/xbm/data/xbm/BasicSR/BasicSR_wave/experiments/baseline/Large/DVD/IMG_0030'
# #----------------------------------------------------------------------------
# sequence = sorted(os.listdir(img_root))
# p_length = 21
# profile = np.zeros(shape = (p_length, img_height, 3))
# print("Temporal Profile Shape:", profile.shape)
# print("No. of frames:", p_length)
# #----------------------------------------------------------------------------
# i = 0
# for image in sequence:
#     if i >= 70 and i <= 90:
#         frame = cv.imread(os.path.join(img_root, image))
#         line = frame[:,w_point,:]
#         profile[i-70,:,:] = line
#     i += 1
# #----------------------------------------------------------------------------
# cv.imwrite('temporalProfile_ours.png', profile)
# print("Profile saved to disk")



#----------------------------------------------------------------------------
img_root = '/media/xbm/data/xbm/cvpr2023/DVD/EDVR/IMG_0030'
#----------------------------------------------------------------------------
sequence = sorted(os.listdir(img_root))
p_length = 21
profile = np.zeros(shape = (p_length, img_height, 3))
print("Temporal Profile Shape:", profile.shape)
print("No. of frames:", p_length)
#----------------------------------------------------------------------------
i = 0
for image in sequence:
    if i >= 70 and i <= 90:
        frame = cv.imread(os.path.join(img_root, image))
        line = frame[:,w_point,:]
        profile[i-70,:,:] = line
    i += 1
#----------------------------------------------------------------------------
cv.imwrite('temporalProfile_edvr.png', profile)
print("Profile saved to disk")



# #----------------------------------------------------------------------------
# img_root = '/media/xbm/data/VideoDeblur_Dataset/DVD/DVD/test/blur/IMG_0030'
# #----------------------------------------------------------------------------
# sequence = sorted(os.listdir(img_root))
# p_length = 21
# profile = np.zeros(shape = (p_length, img_height, 3))
# print("Temporal Profile Shape:", profile.shape)
# print("No. of frames:", p_length)
# #----------------------------------------------------------------------------
# i = 0
# for image in sequence:
#     if i >= 70 and i <= 90:
#         frame = cv.imread(os.path.join(img_root, image))
#         line = frame[:,w_point,:]
#         profile[i-70,:,:] = line
#     i += 1
# #----------------------------------------------------------------------------
# cv.imwrite('temporalProfile_blur.png', profile)
# print("Profile saved to disk")
#
#
#
# #----------------------------------------------------------------------------
# img_root = '/media/xbm/data/VideoDeblur_Dataset/DVD/DVD/test/gt/IMG_0030'
# #----------------------------------------------------------------------------
# sequence = sorted(os.listdir(img_root))
# p_length = 21
# profile = np.zeros(shape = (p_length, img_height, 3))
# print("Temporal Profile Shape:", profile.shape)
# print("No. of frames:", p_length)
# #----------------------------------------------------------------------------
# i = 0
# for image in sequence:
#     if i >= 70 and i <= 90:
#         frame = cv.imread(os.path.join(img_root, image))
#         line = frame[:,w_point,:]
#         profile[i-70,:,:] = line
#     i += 1
# #----------------------------------------------------------------------------
# cv.imwrite('temporalProfile_gt.png', profile)
# print("Profile saved to disk")