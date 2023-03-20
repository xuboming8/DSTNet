import os
import cv2
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

restor_root = '/media/xbm/data/xbm/BasicSR/BaiscSR1/results/GOPRO'
gt_root = '/media/xbm/data/VideoDeblur_Dataset/GOPRO/GOPRO_oriname/test/gt'

psnr = 0.0
ssim = 0.0
count = 0
for video in sorted(os.listdir(restor_root)):
    video_gt_root = gt_root + '/' + video
    video_restor_root = restor_root + '/' + video
    img_gt_list = sorted(os.listdir(video_gt_root))
    img_restor_list = sorted(os.listdir(video_restor_root))
    list_id = 0
    for img in img_gt_list:
        img_gt_root = video_gt_root + '/' + img_gt_list[list_id]
        img_restor_root = video_restor_root + '/' + img_restor_list[list_id]
        img_gt = cv2.imread(img_gt_root)[:, :, ::-1]
        img_restor = cv2.imread(img_restor_root)[:, :, ::-1]
        psnr_ = calculate_psnr(img_restor, img_gt, crop_border=0)
        ssim_ = calculate_ssim(img_restor, img_gt, crop_border=0)
        psnr += psnr_
        ssim += ssim_
        count += 1
        list_id += 1
        print(count, psnr_, ssim_)

print('PSNR:', psnr / count)
print('SSIM:', ssim / count)
