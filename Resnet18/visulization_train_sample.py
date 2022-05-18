from re import L, X

from matplotlib.pyplot import savefig, show
from resnet18 import *
import torchvision.transforms as transforms
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from dataset import cifar100_dataset,Cutout
import numpy as np

def tensor2img(img_arrary):
    img_arrary = img_arrary.numpy()
    r = img_arrary[0] * 255
    g = img_arrary[1] * 255
    b = img_arrary[2] * 255

    ir = Image.fromarray(r).convert("L")
    ig = Image.fromarray(g).convert("L")
    ib = Image.fromarray(b).convert("L")

    im = Image.merge("RGB",(ir,ig,ib))
    return im

def show_img(img_arrary):
    im = tensor2img(img_arrary)
    im.show()
    
def save_img(img_arrary,fp):
    im = tensor2img(img_arrary)
    im.save("img/"+fp)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


transform = transforms.Compose([transforms.ToTensor()])
cifar100_training = torchvision.datasets.CIFAR100(root="data", train=True, download=True,transform = transform)
trainloader = torch.utils.data.DataLoader(cifar100_training, batch_size = 3, shuffle=True, num_workers=0)
cutout = Cutout(length = 8)
_, data = next(enumerate(trainloader,0))

x = data[0]
img1 = x[0]
img2 = x[1]
img3 = x[2]

save_img(img1,"img1.png")
save_img(img2,"img2.png")
save_img(img3,"img3.png")


label1 = data[0][0]
label2 = data[0][1]
label3 = data[0][2]

#cutout
cutimg1 = cutout(img1)
cutimg2 = cutout(img2)
cutimg3 = cutout(img3)

save_img(cutimg1,"cutimg1.png")
save_img(cutimg2,"cutimg2.png")
save_img(cutimg3,"cutimg3.png")


#mixup
index = [1,2,0]
lam   = np.random.beta(0.5, 0.5)
mixed_x = x.clone()
mixed_x = lam * mixed_x + (1 - lam) * mixed_x[index, :]

mixed_img1 = lam * img1 + (1-lam) * img2
mixed_img2 = lam * img2 + (1-lam) * img3
mixed_img3 = lam * img3 + (1-lam) * img1

save_img(mixed_img1,"mixed_img1.png")
save_img(mixed_img2,"mixed_img2.png")
save_img(mixed_img3,"mixed_img3.png")



#cutmix
bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
cutmix_x = x.clone()
cutmix_x[:, :, bbx1:bbx2, bby1:bby2] = cutmix_x[index, :, bbx1:bbx2, bby1:bby2]
cutmix_img1 = cutmix_x[0]
cutmix_img2 = cutmix_x[1]
cutmix_img3 = cutmix_x[2]

save_img(cutmix_img1,"cutmix_img1.png")
save_img(cutmix_img2,"cutmix_img2.png")
save_img(cutmix_img3,"cutmix_img3.png")
