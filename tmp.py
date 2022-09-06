from models import Generator, Colorizer, EncoderF
import pickle
import torch
from torch_ema import ExponentialMovingAverage
from representation import RGBuvHistBlock

model = EncoderF()
print(model)

exit()

with open('./pretrained/config.pickle', 'rb') as f:
    config = pickle.load(f)
G_gt = Generator(**config)
# G_gt.load_state_dict(torch.load('./ckpts/ablation_gfix/EG_009.ckpt',
G_gt.load_state_dict(torch.load('./pretrained/G_ema_256.pth',
                           map_location='cpu'),
                           strict=False)

G_gt.float()
EG = Colorizer(config, 
               './pretrained/G_ema_256.pth', 
               'adabatch')
EG.float()

EG.forward(torch.randn(4,1,256,256),torch.LongTensor([1,2,3,4]), torch.randn(4,119))

# for i in G_gt.named_buffers():
#     print(i[1].requires_grad)
#
# exit()

# for a, b in zip(G_gt.named_parameters(), EG.G.named_parameters()):
#     assert a[0] == b[0]
#     if (a[1].equal(b[1])):
#         print('PASS:', a[0])
#     else:
#         print('FAIL:', a[0])

for a, b in zip(G_gt.named_buffers(), EG.G.named_buffers()):
    assert a[0] == b[0]
    if (a[1].equal(b[1])):
        pass
        # print('PASS:', a[0])
    else:
        print('FAIL:', a[0])
