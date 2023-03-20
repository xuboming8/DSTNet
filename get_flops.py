from basicsr.archs.deblur_arch import Deblur
from basicsr.archs.deblurL_arch import Deblur_L
# from basicsr.archs.edvr_arch import EDVR
# from scripts.metrics.calculate_flops import get_model_flops
from thop import profile
from ptflops import get_model_complexity_info
import torch
from basicsr.archs.basicvsr_pp_arch import BasicVSRPlusPlus
# from basicsr.archs.network_rvrt import RVRT

net = Deblur(num_feat=64, num_block=15).cuda()
# net = Deblur_L(num_feat=64, num_block=30).cuda()
# net = Deblur(num_feat=64, num_block=15, spynet_path='experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth').cuda()
# net = BasicVSRPlusPlus(num_blocks=15, spynet_pretrained='experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth').cuda()
# net = RVRT(upscale=1).cuda()
input_dim = torch.randn(1, 10, 3, 256, 256).cuda()
flops, params = profile(model=net, inputs=(input_dim, ))
print(flops/10**9, params/10**6)

# net = Deblur(num_feat=64, num_block=20, spynet_path='experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth').cuda()
# macs, params = get_model_complexity_info(net, (10, 3, 256, 256), as_strings=True, print_per_layer_stat=True, verbose=True)
# print('{:<30} {:<8}'.format('Computational complexity:', macs))
# print('{:<30} {:<8}'.format('Number of patameters:', params))

# from basicsr.archs.deblur_arch import Deblur
# from fvcore.nn import flop_count_str, flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
# import torch
#
# model = Deblur(num_feat=64, num_block=10).cuda()
#
# x = torch.randn(1, 10, 3, 256, 256).cuda()
# flops = flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x))
# print(flops)