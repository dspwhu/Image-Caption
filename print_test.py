#coding:utf8
import torch as t
from torch.autograd import Variable
import torchvision as tv

from model import CaptionModel
from config import Config

from PIL import Image

opt = Config()
opt.caption_data_path = 'caption.pth' # 原始数据
opt.test_img = 'img/example.jpeg' # 输入图片
opt.use_gpu = False  # 是否使用GPU(没必要)
opt.model_ckpt='caption_0914_1947' # 预训练的模型

# 数据预处理
data = t.load(opt.caption_data_path)
word2ix,ix2word = data['word2ix'],data['ix2word']

IMAGENET_MEAN =  [0.485, 0.456, 0.406]
IMAGENET_STD =  [0.229, 0.224, 0.225]
normalize =  tv.transforms.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD)
transforms = tv.transforms.Compose([
            tv.transforms.Scale(opt.scale_size),
            tv.transforms.CenterCrop(opt.img_size),
            tv.transforms.ToTensor(),
            normalize
    ])
img_ = Image.open(opt.test_img)
img = transforms(img_).unsqueeze(0)
img_.resize((int(img_.width*256/img_.height),256))

# 用resnet50来提取图片特征
# 如果本地没有预训练的模型文件，会自动下载
resnet50 = tv.models.resnet50(True).eval()
del resnet50.fc
resnet50.fc = lambda x:x
if opt.use_gpu:
    resnet50.cuda()
    img = img.cuda()
img_feats = resnet50(Variable(img,volatile=True))


# Caption模型
model = CaptionModel(opt,word2ix,ix2word)
model = model.load(opt.model_ckpt).eval()
if opt.use_gpu:
     model.cuda()

results = model.generate(img_feats.data[0])
print('\r\n'.join(results))