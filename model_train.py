import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # 必须在 import torch 之前
import numpy as np
from torchvision.datasets import ImageFolder 
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as data
import matplotlib.pylab as plt
import matplotlib.ticker
from model import resnet,residual
import torch
torch.backends.cudnn.benchmark = True
from torch import nn
import copy
import time
import pandas as pd

def train_val_data_process():
    train_data = ImageFolder(root = "./data1/train",
                             transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]))
    
    train_data, val_data = data.random_split(train_data,[round(0.8*len(train_data)),round(0.2*len(train_data))])
    train_dataloader = data.DataLoader(dataset=train_data,
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=0)
    val_dataloader = data.DataLoader(dataset=val_data,
                                     batch_size=128,
                                     shuffle=True,
                                     num_workers=0)
    return train_dataloader, val_dataloader

def train_model_process(model,train_dataloader,val_dataloader,num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []
    now = time.time()

    for epoch in range(num_epochs):
        print("epoch {}/{}".format(epoch+1,num_epochs))
        print("-"*20)

        train_loss=0.0
        train_corrects=0
        val_loss=0.0
        val_corrects=0
        train_num=0
        val_num=0

        model.train()
        for step,(b_x,b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            output=model(b_x)
            pre_lab=torch.argmax(output,dim=1)
            loss=criterion(output,b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*b_x.size(0)
            train_corrects += torch.sum(pre_lab==b_y.data)

            train_num += b_x.size(0)

        model.eval()
        for step,(b_x,b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
        
            output = model(b_x)
            pre_lab = torch.argmax(output,dim=1)
            loss = criterion(output,b_y)
            val_loss+=loss.item()*b_x.size(0)
            val_corrects += torch.sum(pre_lab==b_y)
            val_num += b_x.size(0)

        train_loss_all.append(train_loss/train_num)
        train_acc_all.append(train_corrects.double().item()/train_num)
        val_loss_all.append(val_loss/val_num)
        val_acc_all.append(val_corrects.double().item()/val_num)
        print("Epoch: {}/{} " \
        "train_loss: {:.4f}   train_acc: {:.4f} " \
        "val_loss: {:.4f}   val_acc: {:.4f}".format(epoch+1,num_epochs,
                                                        train_loss_all[-1],train_acc_all[-1],
                                                        val_loss_all[-1],val_acc_all[-1]))
        
        print("\n")
        print("time:  {}".format(time.time()-now))
        print("\n")

        if val_acc_all[-1]>best_acc:
            best_acc=val_acc_all[-1]
            best_model_wts=copy.deepcopy(model.state_dict())

    torch.save(best_model_wts,"D:/proj/pyproject/asl_by_resnet/best_model.pth")   

    train_process=pd.DataFrame({"Epoch":list(range(1,num_epochs+1)),
                                "train_loss":train_loss_all,
                                "train_acc":train_acc_all,
                                "val_loss":val_loss_all,
                                "val_acc":val_acc_all})
    return train_process   

def mmap(train_process):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    epoch=np.arange(1,len(train_process)+1)
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
    ax1.plot(epoch,train_process["train_loss"],"rs-",label="train_loss")
    ax1.plot(epoch,train_process["val_loss"],"bs-",label="val_loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3,linestyle="--")
    ax1.set_xticks(epoch)


    ax2.plot(epoch,train_process["train_acc"],"ro-",label="train_acc")
    ax2.plot(epoch,train_process["val_acc"],"bo-",label="val_acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Acc")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3,linestyle="--")
    ax2.set_xticks(epoch)

    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))

    best_ep = int(np.argmax(train_process["val_acc"].values) + 1)
    best_acc = train_process["val_acc"].values.max()
    ax2.plot(best_ep, best_acc, "g*", markersize=16,
             label="best: epoch %d (%.2f%%)" % (best_ep, best_acc * 100))
    ax2.legend()

    fig.suptitle("Training History", fontsize=14)
    fig.tight_layout()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_curve.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("训练曲线已保存到:", out)

    plt.show()


if __name__=="__main__":
    mynet=resnet(residual)
    train_dataloader,val_dataloader=train_val_data_process()
    train_process=train_model_process(mynet,train_dataloader,val_dataloader,num_epochs=20)
    mmap(train_process)
