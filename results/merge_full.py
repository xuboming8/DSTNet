import os
import shutil

# setting
dataset_name = 'GOPRO'
restoration_name = 'GOPRO_test'
root_ori = '/media/xbm/data/VideoDeblur_Dataset/GOPRO/GOPRO_oriname/test/gt/'
test_frame = 10

root = './' + restoration_name + '/visualization/' + dataset_name
now_video = 0
id = 0
frame_len = 0

for clip in sorted(os.listdir(root)):
    video = clip.split("-")[0]
    copy_folder = './' + dataset_name + '/' + video
    if not os.path.exists(copy_folder):
        os.makedirs(copy_folder)

    if now_video != video:
        id = 0
        now_video = video
        frame_len = len(os.listdir(root_ori + video))

    img_path = root + '/' + clip
    if id + test_frame > frame_len:
        miuns = test_frame - (frame_len - id)
        fake_startid = id + 1 + miuns
        for img in sorted(os.listdir(img_path)):
            id = id + 1
            if id < fake_startid:
                pass
            else:
                ori_path = img_path + '/' + img
                copy_path = copy_folder + '/' + str(id - 1 - miuns).zfill(6) + img[-4:]
                shutil.copy(ori_path, copy_path)
    else:
        for img in sorted(os.listdir(img_path)):
            id = id + 1
            ori_path = img_path + '/' + img
            copy_path = copy_folder + '/' + str(id - 1).zfill(6) + img[-4:]
            # print(ori_path, copy_path)
            shutil.copy(ori_path, copy_path)

shutil.rmtree('./' + restoration_name)

