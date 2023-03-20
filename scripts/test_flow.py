import os
import cv2
import torch
from basicsr.archs.raft_arch import RAFT
from basicsr.archs.spynet_arch import SpyNet
from basicsr.archs.arch_util import flow_warp


raft = RAFT(True, 'experiments/pretrained_models/flownet/raft-small.pth').cuda()
spynet = SpyNet('experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth').cuda()
vedio_list = []
i = 0
for filename in sorted(os.listdir(r"/media/xbm/data/VideoDeblur_Dataset/GOPRO/GOPRO1/test/gt/GOPR0384_11_00")):
    print(filename)
    i = i + 1
    if i < 500:
        img = cv2.imread('/media/xbm/data/VideoDeblur_Dataset/GOPRO/GOPRO1/test/gt/GOPR0384_11_00/' + filename)
        img = img / 255.0
        vedio_list.append(torch.from_numpy(img).float().cuda())
    else:
        break
vedio = torch.stack(vedio_list, dim=0)
vedio = vedio.permute(0, 3, 1, 2)   # t c h w

print(vedio.shape)
t, c, h, w = vedio.size()

Max = 0.0
Min = 1280.0
for i in range(0, t - 1):
    frame1 = vedio[i,:,:,:].unsqueeze(dim=0)
    frame2 = vedio[i+1,:,:,:].unsqueeze(dim=0)
    flow = torch.abs(raft(frame1, frame2, iters=12).view(2, h, w))
    Max = max(Max, torch.max(flow))
    Min = min(Min, torch.min(flow))
print("Max:", Max,"Min:", Min)

# vedio_1 = vedio[:-1, :, :, :].cuda()
# vedio_2 = vedio[1:, :, :, :].cuda()
# print(vedio_1.shape, vedio_2.shape)
# flow = raft(vedio_2, vedio_1, iters=12).view(t - 1, 2, h, w).cuda()
# # flow = spynet(vedio_2, vedio_1).view(t - 1, 2, h, w).cuda()
# print(flow.shape)