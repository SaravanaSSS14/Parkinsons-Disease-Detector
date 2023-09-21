from flask import Blueprint , render_template,request
import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

spiral_root = 'C:/Users/nullp/Downloads/Summer_project/spiral'
wave_root = 'C:/Users/nullp/Downloads/Summer_project/wave'

spiral_train = 'C:/Users/nullp/Downloads/Summer_project/spiral/training'
wave_train = 'C:/Users/nullp/Downloads/Summer_project/wave/training'

spiral_test = 'C:/Users/nullp/Downloads/Summer_project/spiral/testing'
wave_test  = 'C:/Users/nullp/Downloads/Summer_project/wave/testing'

size = (256,256)

train_transformations = transforms.Compose([
    transforms.Resize(size),
    transforms.Grayscale(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
])

test_transformations = transforms.Compose([
    transforms.Resize(size),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

spiral_training_dataset = ImageFolder( root = spiral_train, transform = train_transformations, )
wave_training_dataset = ImageFolder( root = wave_train, transform = train_transformations, )

spiral_testing_dataset = ImageFolder( root = spiral_test, transform = test_transformations, )
wave_testing_dataset = ImageFolder( root = wave_test, transform = test_transformations, )

class Separable_Conv(nn.Module):

    def __init__(self, stride, input_channels, output_channels, normalization, activation, res_connect = False):
        
        # initializing variables
        super(Separable_Conv, self).__init__()
        self.stride = stride
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.res_connect = res_connect

        # making the network
        self.conv_layer = nn.Sequential(
            nn.Conv2d( in_channels = self.input_channels, out_channels = 1, kernel_size =3, stride = self.stride, padding = 1),
            nn.Conv2d( in_channels = 1, out_channels = self.output_channels, kernel_size =1, stride = 1, padding =0),
            normalization,
            activation,
        )
    
    def forward(self,input):

        # passing through the network
        output = self.conv_layer(input)

        # making a res connection , if reconnect is true
        if self.res_connect and (self.output_channels == self.input_channels) and (self.stride == 1):
            output = output + input
        
        return output

class Condensed_Conv(nn.Module):

    def __init__(self, stride, input_channels, output_channels, hidden_channels, normalization, activation, res_connect = False):
        
        # initializing variables
        super(Condensed_Conv, self).__init__()
        self.stride = stride
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.res_connect = res_connect
        self.hidden_channels = hidden_channels

        # making the network
        self.conv_layer = nn.Sequential(
            nn.Conv2d( in_channels = self.input_channels, out_channels = self.hidden_channels, kernel_size =3, stride = self.stride, padding = 1),
            nn.Conv2d( in_channels = self.hidden_channels, out_channels = self.output_channels, kernel_size =3, stride = 1, padding = 1),
            normalization,
            activation,
        )
    
    def forward(self,input):

        # passing through the network
        output = self.conv_layer(input)
        
        # making a res connection , if reconnect is true
        if self.res_connect and (self.output_channels == self.input_channels) and (self.stride == 1) :
            output = output + input
        
        return output

class Model1(nn.Module):

    def __init__(self, input_channels, output_channels, arch, size) :

        # initializing variables
        super(Model1, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.arch = arch

        # getting the initial size of the image
        self.h, self.w = size

        if len(arch) < 3:
            print(f"At least arch of depth of 3 is required for good performance")

        # Now making the model
        self.input_conv_layer = nn.Sequential(
            nn.Conv2d(self.input_channels, self.arch[0], kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(arch[0]),
            nn.LeakyReLU(),
        )

        # Now adding the mid Layers
        mid_layers = nn.ModuleList()

        h,w = self.h, self.w

        for i in range(1, len(arch)):
            temp = Condensed_Conv(stride = 1, input_channels = arch[i-1], output_channels = arch[i], normalization = nn.BatchNorm2d(arch[i]), activation = nn.LeakyReLU(), res_connect = True, hidden_channels=arch[i-1])
            mid_layers.append(temp)

            # getting the value of channels and height and width to calculate the number of neurons in the output layer
            h,w,c = h//2, w//2, arch[i]
        
        self.mid_layers = mid_layers
        
        # defining the output conv layer
        self.output_layer = nn.Sequential(
            
            nn.Flatten(),
            nn.Linear(h*w*c, 256),
            nn.LeakyReLU(),
            nn.Linear(256,128),
            nn.LeakyReLU(),

            nn.Linear(128,self.output_channels),
        )
    
    def forward(self, input):

        # passing the output through all layers
        output = self.input_conv_layer(input)

        for i in self.mid_layers:
            output = i(output)
            output = nn.MaxPool2d(kernel_size=2, stride=2)(output)
        
        # print(output.shape)

        output = self.output_layer(output)

        return output

# %%
arch0 = [8,16,32,64,64,128]
classification_model = Model1(input_channels = 1, output_channels = 1, arch = arch0, size = (256,256) ).to(device)

print(F"Total trainable params in classification model is {sum([p.numel() for p in classification_model.parameters() if p.requires_grad])*1e-6:.3f}M")

# %%
spiral_test_dataloader =  DataLoader(spiral_testing_dataset, batch_size = 6, shuffle = True, drop_last = False)

# %%
class Trainer():

    def __init__(self, model, optimizer, loss, train_dataloader, test_dataloader, optimizer_params, name):

        # Initializing variables
        self.name = name
        self.train = train_dataloader
        self.test = test_dataloader
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer(self.model.parameters(), **optimizer_params)
        self.loss = loss
    
    def data_iterator(self, data, train = False):
            
            # these variables will be used later
            epoch_loss = 0
            epoch_metric = 0

            # for storing the truth and prediction
            truth = []
            pred = []

            for index,batch in enumerate(data):
                self.optimizer.zero_grad()
                # getting the image and its labels and moving it to the device
                image,label = batch
                image = image.to(self.device)
                label = label.to(self.device).to(torch.int64)

                if train:
                    # passing to model
                    prediction = self.model(image)
                    # print(prediction,train)
                else:
                    # passing to model
                    prediction = self.model.eval()(image)
                    # print(prediction, train, 'll')

                # print(label,train)

                # getting the loss
                Loss = self.loss(prediction,label)

                if train:
                    # optimizing the model
                    Loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # adding the epoch loss
                epoch_loss += (Loss.item()*len(label))

                # converting prediction to binary
                # prediction = torch.where( prediction > 0.6, 1.0, 0.0)
                
                # appending the truth to calculate the metrics
                truth.append(label.detach().cpu().numpy())
                pred.append(np.argmax(prediction.detach().cpu().numpy(),axis=1))
            
            # no appending the prediction 
            truth = np.concatenate(truth)
            pred = np.concatenate(pred)

            # calculating the metric
            # print(pred)
            epoch_metric = f1_score(truth, pred)*100
            epoch_loss = epoch_loss/len(truth)

            return epoch_metric, epoch_loss
    def train_model(self, n_epochs, print_freq, save_freq, save_path ):
        
        # to store the losses and its performance
        train_loss_list, test_loss_list = [],[]
        train_metric_list, test_metric_list = [],[]

        # loop obj to get a track of the training progress
        loop_obj = range(n_epochs)

        for e in loop_obj:
            train_metric,train_loss = self.data_iterator(self.train, train = True)
            test_metric,test_loss = self.data_iterator(self.test, train = False)

            # now appending it to the list
            train_loss_list.append(train_loss) 
            train_metric_list.append(train_metric)
            test_loss_list.append(test_loss)
            test_metric_list.append(test_metric)

            # now updating the loop obj to know the progress
            # loop_obj.set_description(f"train loss : {train_loss:.3f} | train metric : {train_metric:.3f} | test loss : {test_loss:.3f} | test metric : {test_metric:.3f} ", refresh = True)

            # printing the data according to print freq
            # if (e+1) % print_freq == 0:
            #    print(f"epoch : {e+1} | train loss : {train_loss:.5f} | train metric : {train_metric:.3f} | test loss : {test_loss:.5f} | test metric : {test_metric:.3f} | ")
            
            # if (e+1) % save_freq == 0:
            #     os.makedirs(save_path, exist_ok=True)
            #     torch.save(self.model.state_dict(), os.path.join(save_path,f"{self.name}_{e}.pt") )
        print(f"\nModel has been trained with {n_epochs} images\n ")
        print(f"Train loss\n{train_loss_list}\n Test loss\n{test_loss_list}\n Train accuracy\n{train_metric_list}\nTest accuracy\n {test_metric_list}\n")
        return [train_loss_list, test_loss_list, train_metric_list, test_metric_list]

lr = 5e-5
batch_size = 2

optimizer_params = {'lr' : lr}

optimizer = torch.optim.Adam
loss = nn.CrossEntropyLoss()

# spiral dataset
spiral_train_dataloader = DataLoader(spiral_training_dataset, batch_size = batch_size, shuffle = True, drop_last = False)
spiral_test_dataloader =  DataLoader(spiral_testing_dataset, batch_size = batch_size, shuffle = True, drop_last = False)

# wave dataset
wave_train_dataloader = DataLoader(wave_training_dataset, batch_size = batch_size, shuffle = True, drop_last = False)
wave_test_dataloader =  DataLoader(wave_testing_dataset, batch_size = batch_size, shuffle = True, drop_last = False)

# %%
arch0 = [32,32,64,64,64]
classification_model = Model1(input_channels = 1, output_channels = 2, arch = arch0, size = (256,256) ).to(device)

print(F"Total trainable params in classification model is {sum([p.numel() for p in classification_model.parameters() if p.requires_grad])*1e-6:.3f} M")

# %%
# trainer  = Trainer(model = classification_model, 
#                    optimizer = optimizer, 
#                    loss = loss, 
#                    train_dataloader = spiral_train_dataloader, 
#                    test_dataloader = spiral_test_dataloader, 
#                    optimizer_params = optimizer_params, 
#                    name = 'spiral')

# # %%
# # params for training
# epochs = 25
# print_freq = 1
# save_freq = 5
# save_path = './saved_models/spiral/'

# history = trainer.train_model(n_epochs = epochs, print_freq = print_freq, save_freq = save_freq, save_path = save_path)


# %%
def tester(model = classification_model, dataset = wave_testing_dataset):

    test_dataloader = DataLoader(dataset, batch_size=1, shuffle = False) 

    # these variables will be used later
    epoch_loss = 0
    epoch_metric = 0

    # for storing the truth and prediction
    truth = []
    l_pred = []

    for index,batch in enumerate(test_dataloader):

        # getting the image and its labels and moving it to the device
        image,label = batch
        image = image.to(device)
        label = label.to(device).to(torch.int64)

        # passing to model
        prediction = torch.softmax(model.eval()(image),dim=1)

        # print(prediction)

        # getting the loss
        Loss = loss(prediction,label)
        
        # adding the epoch loss
        epoch_loss += (Loss.item()*len(label))

        # converting prediction to binary
        # pred = torch.where(prediction > 0.6, 1.0, 0.0)
        pred = torch.argmax(prediction,dim=1)
        
        # appending the truth to calculate the metrics
        truth.append(label.detach().cpu().numpy())
        l_pred.append(pred.detach().cpu().numpy())

        print(f"prediction : {'healthy' if int(pred.item()) == 0 else 'parkinson'} with confidence : {prediction[0][pred.item()].item()*100:.2f} % | ", f"Truth : {'healthy' if int(label.item()) == 0 else 'parkinson'}" )

        img = transforms.ToPILImage()(image.squeeze(0).detach().cpu()).convert('RGB')

        if index > 105:
            break
   
    truth = np.concatenate(truth)
    l_pred = np.concatenate(l_pred)

    # calculating the metric
    epoch_metric = f1_score(truth, l_pred)*100
    epoch_loss = epoch_loss/len(truth)

    # print(truth,pred)

    return [epoch_metric, epoch_loss, truth, l_pred]

# %%
# k = tester(model = classification_model, dataset = spiral_testing_dataset)

# %%
# print(k[0])
# Save the entire model including its parameters
model_path = './classification_model.pth'
# torch.save(classification_model.state_dict(), model_path)
# print(f"Model saved at: {model_path}")
# %%
# loading the model
#classification_model.load_state_dict( torch.load(model_path) )

# %%
# type the path of the image here
#test_image_path = Image.open( input('../Input/spiral/testing/parkinson/V14PE03.png') ).convert('RGB')
#test_image = test_transformations(test_image_path).to(device).unsqueeze(0)

# predicting
#prediction = classification_model.to(device).eval()(test_image)
#pred = torch.argmax(prediction,dim=1)

# print(f"prediction : {'healthy' if int(pred.item()) == 0 else 'parkinson'} with confidence : {prediction[0][pred.item()].item()*100:.2f} % | " )

