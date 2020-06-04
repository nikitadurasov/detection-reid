import torchreid
import torch
import numpy as np

import torch.nn.functional as F
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import clear_output
from matplotlib.ticker import NullLocator

import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchreid.models.build_model(
                    name='resnet50',
                    num_classes=4101,
                    loss='softmax',
                    pretrained=False)

#checkpoint = torch.load('./weights/model.pth.tar-90')
checkpoint = torch.load('./weights/resnet50.pth')
#model.load_state_dict(checkpoint['state_dict'])
model.load_state_dict(checkpoint)
model = model.cuda()

def extract_features(model, inputs):
    fm = model.featuremaps(inputs)
    fm = model.global_avgpool(fm)
    return fm.view(fm.size(0), -1)

def torch_format(img):
    return transforms.ToTensor()(img)

def resize(image, size=416):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def get_patch(image, bbox):
    bbox = bbox.astype('int32')
    return image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]

def normalize(image):
    norm_mean = [0.485, 0.456, 0.406] # imagenet mean
    norm_std = [0.229, 0.224, 0.225] # imagenet std
    return transforms.Normalize(mean=norm_mean, std=norm_std)(image)

def patch_to_features(original_frame, bbox, model):
    
    # prepare path for network
    patch = get_patch(original_frame, bbox)
    patch = torch_format(patch)
    patch = resize(patch, [256, 128])
    
    # forward pass to get features
    patch = normalize(patch).cuda()
    patch_features = extract_features(model, patch.unsqueeze(0))
    patch_features = F.normalize(patch_features, p=2, dim=1)
    
    return patch_features

def build_features(image, bboxes, model):
    return torch.cat([patch_to_features(image, bbox, model) for bbox in bboxes], dim=0)

def euclidean_squared_distance(input1, input2):
    """Computes euclidean squared distance.
    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
    Returns:
        torch.Tensor: distance matrix.
    """
    m, n = input1.size(0), input2.size(0)
    mat1 = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n)
    mat2 = torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat = mat1 + mat2
    distmat.addmm_(1, -2, input1, input2.t())
    return distmat

def update_gallery(gallery, features, threshold=0.006):
    
    dist = euclidean_squared_distance(gallery, features)
    new_features = features[torch.all(dist > threshold, dim=0)]
    
    return torch.cat([gallery, new_features], dim=0)

def plot_bboxes(bboxes, ids):
    
    for (x1, y1, x2, y2, conf, cls_conf, cls_pred), idx in zip(bboxes, ids):

        box_w = x2 - x1
        box_h = y2 - y1

        #color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
        # Create a Rectangle patch
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=0.5, edgecolor='r', facecolor="none")
        plt.gca().add_patch(bbox)
        
        plt.text(
                x1,
                y1,
                s=idx,
                color="white",
                verticalalignment="top",
                bbox={"color": 'r', "pad": 0},
        )

video_name = 'milestone3/MOT16-10-raw.webm'
cap = cv2.VideoCapture(video_name)

frame_number = 0
ret, original_frame = cap.read()

filename = f"outputs/{video_name.split('/')[-1]}_frame_{frame_number}.npy"
bboxes = np.load(filename)

gallery = build_features(original_frame, bboxes, model)
indexes = range(len(gallery))

plt.figure(figsize=(16, 8))

plt.imshow(original_frame)
plot_bboxes(bboxes, indexes)

plt.axis("off")
plt.gca().xaxis.set_major_locator(NullLocator())
plt.gca().yaxis.set_major_locator(NullLocator())

plt.savefig(f"bbox_videos/{video_name.split('/')[-1]}_frame_{frame_number:03d}.jpg", format='jpg')

frame_number += 1

while cap.isOpened():
    
    print(f'Processing frame number {frame_number}')
    
    ret, original_frame = cap.read()
    
    if not ret:
        break
    
    original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        
    filename = f"outputs/{video_name.split('/')[-1]}_frame_{frame_number}.npy"
    bboxes = np.load(filename)
    bboxes[bboxes < 0] = 0 
    
    features = build_features(original_frame, bboxes, model)
    
    if frame_number % 10 == 9:
        gallery = update_gallery(gallery, features)
    
    dist_matrix = euclidean_squared_distance(gallery, features)
    indexes = torch.argmin(dist_matrix, dim=0).cpu().numpy()
    
    #clear_output(wait=True)
    plt.figure(figsize=(16, 8))
    
    plt.imshow(original_frame)
    plot_bboxes(bboxes, indexes)
    
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    
    plt.savefig(f"bbox_videos/{video_name.split('/')[-1]}_frame_{frame_number:03d}.jpg", format='jpg')
    plt.show()
    
    frame_number += 1
