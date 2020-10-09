import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"]="True"

def draw_cam(model, img_path, save_path, transform=None, visheadmap=False):
    img = Image.open(img_path).convert('RGB')
    if transform is not None:
        img = transform(img)
    img = img.unsqueeze(0)
    model.eval()
    x = model.conv1(img)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    features = x                #1x2048x7x7
    print(features.shape)
    output = model.avgpool(x)   #1x2048x1x1
    print(output.shape)
    output = output.view(output.size(0), -1)
    print(output.shape)         #1x2048
    output = model.fc(output)   #1x1000
    print(output.shape)
    def extract(g):
        global feature_grad
        feature_grad = g
    pred = torch.argmax(output).item()
    pred_class = output[:, pred]
    features.register_hook(extract)
    pred_class.backward()
    greds = feature_grad
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(greds, (1, 1))
    pooled_grads = pooled_grads[0]
    features = features[0]
    for i in range(2048):
        features[i, ...] *= pooled_grads[i, ...]
    headmap = features.detach().numpy()
    headmap = np.mean(headmap, axis=0)
    headmap /= np.max(headmap)

    if visheadmap:
        plt.matshow(headmap)
        # plt.savefig(headmap, './headmap.png')
        plt.show()

    img = cv2.imread(img_path)
    headmap = cv2.resize(headmap, (img.shape[1], img.shape[0]))
    headmap = np.uint8(255*headmap)
    headmap = cv2.applyColorMap(headmap, cv2.COLORMAP_JET)
    superimposed_img = headmap*0.4 + img
    cv2.imwrite(save_path, superimposed_img)

if __name__ == '__main__':
     model = models.resnet50(pretrained=True)
     transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
     draw_cam(model, './1.jpg', './cam_1.png', transform=transform, visheadmap=True)
