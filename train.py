# Imports here
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

    parser = argparse.ArgumentParser(description='This program is used to train a machine learning model')

    parser.add_argument("data_dir", action="store", dest="data_dir", type=str, help = "the data directorory for the training data is needed")
    parser.add_argument("--save_dir", action="store", dest="save_dir", default = 'flowers', type=str, help = "the data directorory for saving the checkpoint is needed")
    parser.add_argument("--arch", action="store", dest="arch", default = 'vgg19', choices = ['vgg13', 'vgg16', 'vgg19'], type=str, help = "the pretrained model architecture is needed")
    parser.add_argument("--learning_rate", action="store", dest="learning_rate", default = 0.003, type=float, help = "the learning rate is needed")
    parser.add_argument("--hidden_units", action="store", dest="hidden_units", default = 2, type=int, help = "the number of hidden units (from 612 to 25087) is needed")
    parser.add_argument("--epochs", action="store", dest="epochs", default = 5, type=int, help = "the epochs for the training is needed")
    parser.add_argument("--gpu", action="store_true", dest="gpu_confir", type=str, default = gpu, help = "when given, gpu is needed")
    
    ai_args = parser.parse_args()

    data_dir=ai_args.data_dir
    save_dir=ai_args.save_dir
    arch=ai_args.arch
    learning_rate =ai_args.learning_rate
    hidden_units=ai_args.hidden_units
    epochs=ai_args.epochs
    gpu_confir=ai_args.gpu_confir
    
    
    if hidden_units < 612 or hidden_units > 25087:
        print('the hidden_unites are invalid, please choose a value between 612 and 25087')
        quit()
    
    # Choose and load model
    if arch == vgg19
        model = models.vgg19(pretrained=True)
        
    if arch == vgg13
        model = models.vgg13(pretrained=True)
        
    if arch == vgg16
        model = models.vgg19(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    
    # Load Data
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])



    # Load the datasets with ImageFolder
    training_data = datasets.ImageFolder(train_dir, transform=training_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    testing_data = datasets.ImageFolder(test_dir, transform=testing_transforms)


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(training_data, batch_size=56, shuffle=True)
    # trainloader = torch.utils.data.DataLoader(training_data, batch_size=34)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=56)
    testloader = torch.utils.data.DataLoader(testing_data, batch_size=56)

    import json

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() and gpu_confir else "cpu")
    
    
    
    model.classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_units, math.floor(hidden_units/2)),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(math.floor(hidden_units/2), math.floor(hidden_units/4)),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(math.floor(hidden_units/4), 102),
                                 nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

    model.to(device);

    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # Keep session active
    from workspace_utils import active_session

    with active_session():
    # Train model
        epochs = 5
        steps = 0
        running_loss = 0
        print_every = 5
        for epoch in range(epochs):
            for inputs, labels in trainloader:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)
            
                optimizer.zero_grad()
            
                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
    
                running_loss += loss.item()
                # Validate the model while training
                if steps % print_every == 0:
                    validation_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in validationloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)
                            
                            validation_loss += batch_loss.item()
                    
                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                    print(f"Epoch {epoch+1}/{epochs}.. \n"
                          f"Train loss: {running_loss/print_every:.3f}.. \n"
                          f"validation loss: {validation_loss/len(validationloader):.3f}.. \n"
                          f"validation accuracy: {accuracy/len(validationloader):.3f} \n")
                    running_loss = 0
                    model.train()

    # Validation on the test set
    for inputs, labels in testloader:
        test_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)
                
                test_loss += batch_loss.item()
                    
                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Test loss: {test_loss/len(testloader):.3f}..  \n"
                      f"Test accuracy: {accuracy/len(testloader):.3f} \n")

    model.train()
    # Save the checkpoint
    checkpoint = {'state_dict': model.state_dict(),
              'classifier': model.classifier, 
              'class_to_idx_train': training_data.class_to_idx,
              'class_to_idx_validation': validation_data.class_to_idx,
              'class_to_idx_test': testing_data.class_to_idx
              }
    save_path = save_dir + '/checkpoint_flower.pth'
    torch.save(checkpoint, save_path)

    
    