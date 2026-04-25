import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================

IMAGE_DIR = "/home/drumssheet/Dataset/sheetmusic"
MASK_DIR  = "/home/drumssheet/Dataset/staffline_masks"

IMG_SIZE = 2048
BATCH = 1
EPOCHS = 50
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)

os.makedirs("epoch_outputs", exist_ok=True)
os.makedirs("training_curves", exist_ok=True)

log_file = open("training_log.txt","w")
log_file.write("epoch,loss,accuracy\n")

loss_history = []
acc_history = []

# ==============================
# LETTERBOX (KEEP ASPECT RATIO)
# ==============================

def letterbox(img, size):
    h, w = img.shape[:2]
    scale = size / max(h, w)

    nh, nw = int(h*scale), int(w*scale)
    img_resized = cv2.resize(img,(nw,nh))

    canvas = np.zeros((size,size),dtype=img.dtype)
    canvas[:nh,:nw] = img_resized
    return canvas

def letterbox_color(img, size):
    h, w = img.shape[:2]
    scale = size / max(h, w)

    nh, nw = int(h*scale), int(w*scale)
    img_resized = cv2.resize(img,(nw,nh),interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((size,size,3),dtype=img.dtype)
    canvas[:nh,:nw] = img_resized
    return canvas

# ==============================
# DATASET
# ==============================

class StaffDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = os.listdir(img_dir)

        self.aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        img_path = os.path.join(self.img_dir, name)
        mask_path = os.path.join(self.mask_dir, name)

        img = cv2.imread(img_path,0)
        mask = cv2.imread(mask_path)

        img = letterbox(img, IMG_SIZE)
        mask = letterbox_color(mask, IMG_SIZE)

        augmented = self.aug(image=img, mask=mask)
        img = augmented["image"]
        mask = augmented["mask"]

        # color → class
        class_mask = np.zeros((mask.shape[0],mask.shape[1]),dtype=np.uint8)
        class_mask[np.all(mask==[0,0,0],axis=-1)] = 0
        class_mask[np.all(mask==[255,0,0],axis=-1)] = 1
        class_mask[np.all(mask==[0,0,255],axis=-1)] = 2
        mask = class_mask

        img = img/255.0
        img = np.expand_dims(img,0)

        return torch.tensor(img,dtype=torch.float32), torch.tensor(mask,dtype=torch.long)

# ==============================
# MODEL
# ==============================

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def C(i,o):
            return nn.Sequential(
                nn.Conv2d(i,o,3,1,1),
                nn.ReLU(),
                nn.Conv2d(o,o,3,1,1),
                nn.ReLU()
            )

        self.c1=C(1,64); self.p1=nn.MaxPool2d(2)
        self.c2=C(64,128); self.p2=nn.MaxPool2d(2)
        self.c3=C(128,256); self.p3=nn.MaxPool2d(2)
        self.b=C(256,512)

        self.u3=nn.ConvTranspose2d(512,256,2,2); self.c4=C(512,256)
        self.u2=nn.ConvTranspose2d(256,128,2,2); self.c5=C(256,128)
        self.u1=nn.ConvTranspose2d(128,64,2,2); self.c6=C(128,64)

        self.out=nn.Conv2d(64,3,1)

    def forward(self,x):
        c1=self.c1(x)
        c2=self.c2(self.p1(c1))
        c3=self.c3(self.p2(c2))
        b=self.b(self.p3(c3))

        u3=self.u3(b); u3=torch.cat([u3,c3],1); u3=self.c4(u3)
        u2=self.u2(u3); u2=torch.cat([u2,c2],1); u2=self.c5(u2)
        u1=self.u1(u2); u1=torch.cat([u1,c1],1); u1=self.c6(u1)

        return self.out(u1)

# ==============================
# LOSS
# ==============================

ce_loss = nn.CrossEntropyLoss()

def dice_loss_multiclass(pred, target, smooth=1e-6):
    pred = torch.softmax(pred, dim=1)
    dice = 0
    for cls in [1,2]:
        p = pred[:,cls]
        t = (target==cls).float()
        inter = (p*t).sum()
        union = p.sum()+t.sum()
        dice += (2*inter+smooth)/(union+smooth)
    return 1-(dice/2)

def pixel_acc(pred, mask):
    pred = torch.argmax(pred,dim=1)
    return (pred==mask).float().mean()

# ==============================
# DATA
# ==============================

dataset = StaffDataset(IMAGE_DIR, MASK_DIR)
loader = DataLoader(dataset,batch_size=BATCH,shuffle=True)
print("Total images:",len(dataset))

# ==============================
# MODEL SETUP
# ==============================

model = UNet().to(DEVICE)
opt = torch.optim.Adam(model.parameters(),lr=LR)

best_loss = 999

# ==============================
# TRAIN
# ==============================

for ep in range(EPOCHS):
    model.train()
    total_loss=0
    total_acc=0

    loop=tqdm(loader,desc=f"Epoch {ep+1}/{EPOCHS}")

    for img,mask in loop:
        img,mask=img.to(DEVICE),mask.to(DEVICE)

        pred=model(img)

        loss1=ce_loss(pred,mask)
        loss2=dice_loss_multiclass(pred,mask)
        loss=loss1+0.7*loss2

        opt.zero_grad()
        loss.backward()
        opt.step()

        acc=pixel_acc(pred,mask)

        total_loss+=loss.item()
        total_acc+=acc.item()

        loop.set_postfix(loss=loss.item(),acc=acc.item())

    avg_loss=total_loss/len(loader)
    avg_acc=total_acc/len(loader)

    print(f"Epoch {ep+1} | Loss {avg_loss:.4f} | Acc {avg_acc:.4f}")

    log_file.write(f"{ep+1},{avg_loss:.6f},{avg_acc:.6f}\n")
    log_file.flush()

    loss_history.append(avg_loss)
    acc_history.append(avg_acc)

    if avg_loss<best_loss:
        best_loss=avg_loss
        torch.save(model.state_dict(),"best_model.pth")
        print("Saved BEST model")

    # ===== save curves =====
    plt.figure(); plt.plot(loss_history); plt.title("Loss")
    plt.savefig("training_curves/loss.png"); plt.close()

    plt.figure(); plt.plot(acc_history); plt.title("Accuracy")
    plt.savefig("training_curves/accuracy.png"); plt.close()

    # ===== save output =====
    model.eval()
    test_img=os.listdir(IMAGE_DIR)[0]
    img=cv2.imread(os.path.join(IMAGE_DIR,test_img),0)
    orig=letterbox(img,IMG_SIZE)

    img_r=orig/255.0
    img_t=torch.tensor(img_r).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        p=model(img_t)
        p=torch.argmax(p,1)[0].cpu().numpy()

    vis=np.zeros((IMG_SIZE,IMG_SIZE,3),dtype=np.uint8)
    vis[p==0]=[255,255,255]
    vis[p==1]=[0,0,255]
    vis[p==2]=[0,255,0]

    cv2.imwrite(f"epoch_outputs/epoch_{ep+1}.png",vis)

torch.save(model.state_dict(),"final_model.pth")
log_file.close()

print("Training complete")
