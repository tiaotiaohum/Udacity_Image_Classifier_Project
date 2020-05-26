# Import necessary modules
import argparse
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import math
import numpy as np
from PIL import Image
import math

import matplotlib.pyplot as plt

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

def main():
    # parse arguments from command line to .py file
    parser = argparse.ArgumentParser(description='This program is used to train a machine learning model')
    
    parser.add_argument("checkpoint", action="store", dest="checkpoint", default = "flowers/checkpoint_flower.pth", type=str, help = "the checkpoint directorory (path and filename)")
    parser.add_argument("--top_k", action="store", dest="top_classes", default = 'flowers', type=int, help = "K most likely classes")
    parser.add_argument("--category_names", action="store", dest="category_names", default = 'cat_to_name.json', type=str, help = "the file, which is used to map the category to real name")
    parser.add_argument("--gpu", action="store_true", dest="gpu_confir", type=str, default = gpu, help = "when given, gpu is needed")
    ai_args = parser.parse_args()

    # assign name to easy use
    top_classes=ai_args.top_classes
    checkpoint=ai_args.checkpoint
    category_names=ai_args.category_names
    gpu_confir=ai_args.gpu_confir
    
    # load the file for mapping the category to real name
    import json

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    # load the saved checkpoint, which contains the trained model
    def load_checkpoint(path=checkpoint):
        if gpu_confir:
            device = 'gpu'
        else:
            device = 'cpu'
        checkpoint = torch.load(path, device)
        model = models.vgg19(pretrained=True)
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        class_to_idx_train = checkpoint['class_to_idx_train']
        class_to_idx_validation = checkpoint['class_to_idx_validation']
        class_to_idx_test = checkpoint['class_to_idx_test']
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
        
        return model, class_to_idx_train
    model, class_to_idx_train = load_checkpoint('checkpoint_flower.pth')
    
    # Process a PIL image for use in a PyTorch model
    def process_image(image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        im = Image.open(image)
    
        # Resize the image
        width, height = im.size
        if width > height:
            ratio = float(width) / float(height)
            newwidth = ratio * 256
            image = im.resize((math.floor(newwidth), 256))
        else:
            ratio = float(height) / float(width)
            newheight = ratio * 256
            image = im.resize((256, math.floor(newheight)))
    
        # Crop the image
        width_new, height_new = image.size
        left = (width_new-224)/2
        upper = (height_new-224)/2
        right = (width_new+224)/2
        lower = (height_new+224)/2
        image = image.crop((left, upper, right, lower)) 
 
        # Normalize the image
        np_image = np.array(image)
        np_image = np_image/255
        mean = np.array([0.485, 0.456, 0.406])
        standard_deviation = np.array([0.229, 0.224, 0.225])
        normalized_np_image = (np_image-mean)/standard_deviation
        transposed_np_image = normalized_np_image.transpose((2,0,1))
        transposed_np_image = torch.from_numpy(transposed_np_image).type(torch.FloatTensor)

        return transposed_np_image
    
    def imshow(image, ax=None, title=None):
        """Imshow for Tensor."""
        if ax is None:
            fig, ax = plt.subplots()
        
        # PyTorch tensors assume the color channel is the first dimension
        # but matplotlib assumes is the third dimension
        image = image.numpy().transpose((1, 2, 0))
        
        # Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        
        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)
    
        ax.imshow(image)
    
        return ax
    
    # Prediction 
    def predict(image_path, model, topk, class_to_idx_train):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        
        # Predict the class from an image file
        process_inputs = process_image(image_path)

        process_inputs.unsqueeze_(0)

        with torch.no_grad():
            model.eval()
            logps = model.forward(process_inputs)
            ps = torch.exp(logps)
            top_p, top_class = torch.topk(ps, topk)

        top_p =top_p.numpy()
        top_class = top_class.numpy()
        idx_to_class = {class_to_idx_train[i]: i for i in class_to_idx_train}
    
        pred_classes = list()

        for top_class in top_class[0]:
            pred_classes.append(idx_to_class[top_class])
        
        return top_p[0], pred_classes
    
    # Display an image along with the required n top_classes
    image_path = 'flowers/train/82/image_01644.jpg'
    top_p, classes = predict(image_path, model, top_classes, class_to_idx_train)

    name_list = []
    for classes in classes:
        name_list.append(cat_to_name[classes])

    def plotgraph(image_path, top_p, name_list):
        correct_name = cat_to_name['82']
        image = Image.open(image_path)
        fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=1, nrows=2)
        # first plot
        ax1 = plt.subplot(211)
        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title(correct_name)
        # second plot
        num_list = [top_p[0], top_p[1], top_p[2], top_p[3], top_p[4]]
        num_list.reverse()

        name_list = tuple(name_list)
        name_list_rev = name_list[::-1]
        y_pos = np.arange(len(name_list))
        ax2.barh(range(len(num_list)), num_list)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(name_list_rev)
        plt.tight_layout()

    
    plotgraph(image_path, top_p, name_list)