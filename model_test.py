import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # 必须在 import torch 之前
from model import googlenet,inception
from torchvision.datasets import ImageFolder
import torch.utils.data as data
import torch
torch.backends.cudnn.benchmark = True
from data_utils import get_transforms, BASE_DIR, CLASSES
from PIL import Image
from torchvision import transforms

def test_data_process():
    # 验证集：与训练相同的 Resize -> ToTensor -> Normalize（整体均值/方差）
    test_data=ImageFolder(root="D:/proj/pyproject/dog_vs_cat_by_googlenet/data/val",
                          transform=get_transforms())
    test_dataloader=data.DataLoader(dataset=test_data,
                                    batch_size=1,
                                    shuffle=True,
                                    num_workers=0)
    return test_dataloader

def test_model_process(model,test_dataloader):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    test_corrects=0
    test_num=0
    model.eval()
    with torch.no_grad():
        for test_data_x,test_data_y in test_dataloader:
            test_data_x=test_data_x.to(device)
            test_data_y=test_data_y.to(device)
            output=model(test_data_x)
            pre_lab=torch.argmax(output,dim=1)
            test_corrects+=torch.sum(pre_lab==test_data_y.data)
            test_num+=test_data_x.size(0)
            test_acc=test_corrects.double().item()/test_num
    print("test_acc:{}".format(test_acc))

if __name__=="__main__":
    model=googlenet(inception)
    model.load_state_dict(torch.load("best_model.pth"))
    # test_dataloader=test_data_process()
    # # test_model_process(model,test_dataloader)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # with torch.no_grad():
    #     model.eval()
    #     for b_x,b_y in test_dataloader:
    #         b_x = b_x.to(device)
    #         b_y = b_y.to(device)
    #         output = model(b_x)
    #         pre_lab = torch.argmax(output,dim=1)
    #         result =pre_lab.item()
    #         label = b_y.item()
    #         print("pre_lab:{}    ,   true_lab:{}".format(CLASSES[result],CLASSES[label]))

    image = Image.open("D:/proj/pyproject/dog_vs_cat_by_googlenet/OIP-C (2).jpg")
    normalize = transforms.Normalize([0.488,0.455,0.417],[0.260,0.253,0.256])
    train_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),normalize])
    image = train_transforms(image)
    image = image.unsqueeze(0)
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        image = image.to(device)
        output = model(image)
        pre_lab = torch.argmax(output,dim=1)
        pre_lab = pre_lab.item()
        print("i think it is {}".format(CLASSES[pre_lab]))

