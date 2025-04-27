# In The Name of God
##### Kolmogorov–Arnold PointNet: Deep learning for prediction of fluid fields on irregular geometries #####

#Author: Ali Kashefi (kashefi@stanford.edu)

##### Citation #####
#If you use the code, please cite the following journal paper: 
#[Kolmogorov–Arnold PointNet: Deep learning for prediction of fluid fields on irregular geometries]
#(https://doi.org/10.1016/j.cma.2025.117888)

#@article{kashefi2024KANpointnetCFD,
#title = {Kolmogorov–Arnold PointNet: Deep learning for prediction of fluid fields on irregular geometries},
#journal = {Computer Methods in Applied Mechanics and Engineering},
#volume = {439},
#pages = {117888},
#year = {2025},
#doi = {https://doi.org/10.1016/j.cma.2025.117888},
#author = {Ali Kashefi}}


###### Libraries ######
import time 
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
import os
import linecache
from operator import itemgetter
import numpy as np
from numpy import zeros
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = '12'
from mpl_toolkits.mplot3d import Axes3D

###### Device setup ######
if torch.cuda.is_available():
    device = torch.device("cuda")  # GPU available
else:
    device = torch.device("cpu")  # Only CPU available

torch.manual_seed(0)

###### Parameter setup ######
Degree = 3 # Polynomial degree of Jacaboi Polynomial
Alpha = -0.5 # \alpha in Jacaboi Polynomial
Beta = -0.5 # \beta in Jacaboi Polynomial
Scaling = 1.0 # To control the network size
num_epochs = 5000 # Maximum numer of epochs for training
Batch_Size_Train = 128 # batch size in training set
Batch_Size_Validation = 100 # batch size in validation set

###### Functions for data visualization and error calculations ######

def plot2DPointCloud(x_coord,y_coord,file_name):
    plt.scatter(x_coord,y_coord,s=2.5)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    x_upper = np.max(x_coord) + 1
    x_lower = np.min(x_coord) - 1
    y_upper = np.max(y_coord) + 1
    y_lower = np.min(y_coord) - 1
    plt.xlim([x_lower, x_upper])
    plt.ylim([y_lower, y_upper])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(file_name+'.png',dpi=300)
    plt.savefig(file_name+'.eps')
    plt.savefig(file_name+'.pdf',format='pdf')
    plt.clf()
    #plt.show()

def plotSolution(x_coord,y_coord,solution,file_name,title):
    plt.scatter(x_coord, y_coord, s=2.5,c=solution,cmap='jet')
    plt.title(title)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    x_upper = np.max(x_coord) + 1
    x_lower = np.min(x_coord) - 1
    y_upper = np.max(y_coord) + 1
    y_lower = np.min(y_coord) - 1
    plt.xlim([x_lower, x_upper])
    plt.ylim([y_lower, y_upper])
    plt.gca().set_aspect('equal', adjustable='box')
    cbar= plt.colorbar()
    plt.savefig(file_name+'.png',dpi=300)
    plt.savefig(file_name+'.eps')
    plt.savefig(file_name+'.pdf',format='pdf')
    plt.clf()
    #plt.show()

def plot_loss(training_losses, validation_losses):
    plt.plot(training_losses, label='Training Loss', color='blue')
    plt.plot(validation_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig('loss.png', dpi=300)
    #plt.savefig('loss.eps')
    plt.clf()
    #plt.show()

def compute_rms_error(generated_cloud, variable, ground_truth):
    generated = generated_cloud[0, variable, :]
    squared_diff = (generated - ground_truth) ** 2
    mean_squared_diff = np.mean(squared_diff)
    rms_error = np.sqrt(mean_squared_diff)
    return rms_error

def compute_relative_error(generated_cloud, variable, ground_truth):
    generated = generated_cloud[0, variable, :]
    difference = generated - ground_truth
    norm_difference = np.linalg.norm(difference)
    norm_ground_truth = np.linalg.norm(ground_truth)
    relative_error = norm_difference / norm_ground_truth
    return relative_error

def plot_histogram(x,title):
    x = [i * 100 for i in x]
    plt.hist(x, bins='auto', edgecolor='black')
    plt.title(title)
    plt.xlabel("Relative error")
    plt.ylabel("Frequency")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(title+'.png',dpi=300)
    #plt.savefig(file_name+'.eps')
    plt.savefig(title+'.pdf',format='pdf')
    plt.clf()

###### Data loading and data preparation ######
Data = np.load('CFDdata.npy')
data_number = Data.shape[0]

point_numbers = 1024
space_variable = 2 # 2 in 2D (x,y) and 3 in 3D (x,y,z)
cfd_variable = 3 # (u, v, p); which are the x-component of velocity, y-component of velocity, and pressure fields

input_data = zeros([data_number,point_numbers,space_variable],dtype='f')
output_data = zeros([data_number,point_numbers,cfd_variable],dtype='f')

input_data[:, :, 0] = Data[:, :, 0]  # x coordinate
input_data[:, :, 1] = Data[:, :, 1]  # y coordinate

output_data[:, :, 0] = Data[:, :, 3]  # u (m/s)
output_data[:, :, 1] = Data[:, :, 4]  # v (m/s)
output_data[:, :, 2] = Data[:, :, 2]  # p (Pa)

###### Normalize spatial coordinates ######
x_min = np.min(input_data[:,:,0])
x_max = np.max(input_data[:,:,0])
y_min = np.min(input_data[:,:,1])
y_max = np.max(input_data[:,:,1])

input_data[:,:,0] = 2*(input_data[:,:,0] - x_min)/(x_max - x_min) - 1
input_data[:,:,1] = 2*(input_data[:,:,1] - y_min)/(y_max - y_min) - 1

###### Split data ######
training_idx = np.load('training_idx.npy')
validation_idx = np.load('validation_idx.npy')
test_idx = np.load('test_idx.npy')

#all_indices = np.random.permutation(data_number)
#training_idx = all_indices[:int(0.8*data_number)]
#validation_idx = all_indices[int(0.8*data_number):int(0.9*data_number)]
#test_idx = all_indices[int(0.9*data_number):]

input_train, input_validation, input_test = input_data[training_idx,:], input_data[validation_idx,:], input_data[test_idx,:]
output_train, output_validation, output_test = output_data[training_idx,:], output_data[validation_idx,:], output_data[test_idx,:]

##### Normalize the train set and validation set outputs #######
u_min = np.min(output_train[:,:,0])
u_max = np.max(output_train[:,:,0])
v_min = np.min(output_train[:,:,1])
v_max = np.max(output_train[:,:,1])
p_min = np.min(output_train[:,:,2])
p_max = np.max(output_train[:,:,2])

output_train[:,:,0] = 2*(output_train[:,:,0] - u_min)/(u_max - u_min) - 1
output_train[:,:,1] = 2*(output_train[:,:,1] - v_min)/(v_max - v_min) - 1
output_train[:,:,2] = 2*(output_train[:,:,2] - p_min)/(p_max - p_min) - 1

output_validation[:,:,0] = 2*(output_validation[:,:,0] - u_min)/(u_max - u_min) - 1
output_validation[:,:,1] = 2*(output_validation[:,:,1] - v_min)/(v_max - v_min) - 1
output_validation[:,:,2] = 2*(output_validation[:,:,2] - p_min)/(p_max - p_min) - 1

##### Dataset and Network #####
class CreateDataset(Dataset):
    def __init__(self, input_data_x, input_data_y, output_data_u, output_data_v, output_data_p):
        assert input_data_x.shape == input_data_y.shape == output_data_u.shape == output_data_v.shape == output_data_p.shape, \
            "All input and output data must have the same shape."

        self.input_data_x = input_data_x
        self.input_data_y = input_data_y
        self.output_data_u = output_data_u
        self.output_data_v = output_data_v
        self.output_data_p = output_data_p

    def __len__(self):
        return self.input_data_x.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.input_data_x[idx]
        y = self.input_data_y[idx]
        u = self.output_data_u[idx]
        v = self.output_data_v[idx]
        p = self.output_data_p[idx]

        input_data = torch.stack((x, y), dim=0)  # Shape: [2, num_points]
        targets = torch.stack((u, v, p), dim=0)  # Shape: [3, num_points]

        return input_data, targets

###### Shared Kolmogorov-Arnold Networks (KANs) ######
class JacobiKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, a=1.0, b=1.0):
        super(JacobiKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.a = a
        self.b = b
        self.degree = degree

        self.jacobi_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        batch_size, input_dim, num_points = x.shape
        x = x.permute(0, 2, 1).contiguous()  # shape = (batch_size, num_points, input_dim)
        x = torch.tanh(x)  # Normalize x to [-1, 1]

        # Initialize Jacobi polynomial tensors
        jacobi = torch.ones(batch_size, num_points, self.input_dim, self.degree + 1, device=x.device)

        if self.degree > 0:
            jacobi[:, :, :, 1] = ((self.a - self.b) + (self.a + self.b + 2) * x) / 2

        for i in range(2, self.degree + 1):
            A = (2*i + self.a + self.b - 1)*(2*i + self.a + self.b)/((2*i) * (i + self.a + self.b))
            B = (2*i + self.a + self.b - 1)*(self.a**2 - self.b**2)/((2*i)*(i + self.a + self.b)*(2*i+self.a+self.b-2))
            C = -2*(i + self.a -1)*(i + self.b -1)*(2*i + self.a + self.b)/((2*i)*(i + self.a + self.b)*(2*i + self.a + self.b -2))
            jacobi[:, :, :, i] = (A*x + B)*jacobi[:, :, :, i-1].clone() + C*jacobi[:, :, :, i-2].clone()

        # Compute the Jacobi interpolation
        jacobi = jacobi.permute(0, 2, 3, 1)  # shape = (batch_size, input_dim, degree + 1, num_points)
        y = torch.einsum('bids,iod->bos', jacobi, self.jacobi_coeffs)  # shape = (batch_size, output_dim, num_points)

        return y

###### PointNet with shared KANs ######
class PointNetKAN(nn.Module):
    def __init__(self, input_channels, output_channels, scaling=1.0, poly_degree=3, alpha=1.0, beta=1.0):
        super(PointNetKAN, self).__init__()

        #shared KANs (64, 64)
        self.jacobikan1 = JacobiKANLayer(input_channels, int(64 * scaling), poly_degree, alpha, beta)
        self.jacobikan2 = JacobiKANLayer(int(64 * scaling), int(64 * scaling), poly_degree, alpha, beta)

        #shared KANs (64, 128, 1024)
        self.jacobikan3 = JacobiKANLayer(int(64 * scaling), int(64 * scaling), poly_degree, alpha, beta)
        self.jacobikan4 = JacobiKANLayer(int(64 * scaling), int(128 * scaling), poly_degree, alpha, beta)
        self.jacobikan5 = JacobiKANLayer(int(128 * scaling), int(1024 * scaling), poly_degree, alpha, beta)

        #shared KANs (512, 256, 128)
        self.jacobikan6 = JacobiKANLayer(int(1024 * scaling) + int(64 * scaling), int(512 * scaling), poly_degree, alpha, beta)
        self.jacobikan7 = JacobiKANLayer(int(512 * scaling), int(256 * scaling), poly_degree, alpha, beta)
        self.jacobikan8 = JacobiKANLayer(int(256 * scaling), int(128 * scaling), poly_degree, alpha, beta)

        #shared KANs (128, output_channels)
        self.jacobikan9 = JacobiKANLayer(int(128 * scaling), int(128 * scaling), poly_degree, alpha, beta)
        self.jacobikan10 = JacobiKANLayer(int(128 * scaling), output_channels, poly_degree, alpha, beta)

        #batch normalization
        self.bn1 = nn.BatchNorm1d(int(64 * scaling))
        self.bn2 = nn.BatchNorm1d(int(64 * scaling))
        self.bn3 = nn.BatchNorm1d(int(64 * scaling))
        self.bn4 = nn.BatchNorm1d(int(128 * scaling))
        self.bn5 = nn.BatchNorm1d(int(1024 * scaling))
        self.bn6 = nn.BatchNorm1d(int(512 * scaling))
        self.bn7 = nn.BatchNorm1d(int(256 * scaling))
        self.bn8 = nn.BatchNorm1d(int(128 * scaling))
        self.bn9 = nn.BatchNorm1d(int(128 * scaling))

    def forward(self, x):

        # shared KANs (64, 64)
        x = self.jacobikan1(x)
        x = self.bn1(x)
        x = self.jacobikan2(x)
        x = self.bn2(x)

        local_feature = x

        # shared KANs (64, 128, 1024)
        x = self.jacobikan3(x)
        x = self.bn3(x)
        x = self.jacobikan4(x)
        x = self.bn4(x)
        x = self.jacobikan5(x)
        x = self.bn5(x)

        # max pooling to get the global feature
        global_feature = F.max_pool1d(x, kernel_size=x.size(-1))
        global_feature = global_feature.view(-1, global_feature.size(1), 1).expand(-1, -1, num_points)

        # concatenate local and global features
        x = torch.cat([local_feature, global_feature], dim=1)

        # shared KANs (512, 256, 128)
        x = self.jacobikan6(x)
        x = self.bn6(x)
        x = self.jacobikan7(x)
        x = self.bn7(x)
        x = self.jacobikan8(x)
        x = self.bn8(x)

        # shared KANs (128, output_channels)
        x = self.jacobikan9(x)
        x = self.bn9(x)
        x = self.jacobikan10(x)

        return x

###################################################
num_samples = data_number
num_points = 1024

input_train = torch.from_numpy(input_train).float()
input_validation = torch.from_numpy(input_validation).float()
input_test = torch.from_numpy(input_test).float()

output_train = torch.from_numpy(output_train).float()
output_validation = torch.from_numpy(output_validation).float()
output_test = torch.from_numpy(output_test).float()

training_dataset = CreateDataset(input_train[:,:,0],input_train[:,:,1],output_train[:,:,0],output_train[:,:,1],output_train[:,:,2])
validation_dataset = CreateDataset(input_validation[:,:,0],input_validation[:,:,1],output_validation[:,:,0],output_validation[:,:,1],output_validation[:,:,2])

dataloader_Train = DataLoader(training_dataset, batch_size=Batch_Size_Train, shuffle=True, drop_last=True)
dataloader_Validation = DataLoader(validation_dataset, batch_size=Batch_Size_Validation, shuffle=True, drop_last=True)

model = PointNetKAN(input_channels=space_variable, output_channels=cfd_variable, poly_degree=Degree, scaling=Scaling, alpha=Alpha, beta=Beta)
model = model.to(device)

###### calculate the number of trainable parameters ######
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_parameters = count_parameters(model)
print(f'The number of trainable parameters in the model: {num_parameters}')

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

epoch_losses = []
validation_losses = []

total_training_time = 0

###### training ######
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    start_time = time.time() 
 
    for i, (inputs, targets) in enumerate(dataloader_Train):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    end_time = time.time()
    total_training_time += (end_time - start_time)

    print(f'Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {running_loss/len(training_idx):.8f}')
    epoch_losses.append(running_loss)

    # validation loop
    model.eval()
    val_running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader_Validation:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            val_loss = criterion(outputs, targets)
            val_running_loss += val_loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Average Validation Loss: {val_running_loss/len(validation_idx):.8f}')
    validation_losses.append(val_running_loss)

print()
print(f'Total training time: {total_training_time:.2f} seconds')
print()

validation_losses = [loss / len(validation_idx) for loss in validation_losses]
epoch_losses = [loss / len(training_idx) for loss in epoch_losses]

with open('validation_losses.txt', 'w') as file:
    for loss in validation_losses:
        file.write(f"{loss}\n")

with open('epoch_losses.txt', 'w') as file:
    for loss in epoch_losses:
        file.write(f"{loss}\n")

plot_loss(epoch_losses,validation_losses)

###### Error analysis of the training set ######
rms_u, rms_v, rms_p = 0.0, 0.0, 0.0
lrms_u, lrms_v, lrms_p = 0.0, 0.0, 0.0

for j in range(len(training_idx)):
    model.eval()
    input_data_pred = torch.stack((input_train[j,:,0].float(), input_train[j,:,1].float()), dim=0)
    input_data_pred = input_data_pred.unsqueeze(0)

    with torch.no_grad():
        input_data_pred = input_data_pred.to(device)
        predictions = model(input_data_pred) #shape is [1, 3, num_points]

    #back to usual x and y coordinates
    input_train[j,:,0] = (input_train[j,:,0] + 1)*(x_max - x_min)/2 + x_min
    input_train[j,:,1] = (input_train[j,:,1] + 1)*(y_max - y_min)/2 + y_min

    #back to the physical domain
    predictions[0,0,:] = (predictions[0,0,:]+1)*(u_max - u_min)/2 + u_min
    predictions[0,1,:] = (predictions[0,1,:]+1)*(v_max - v_min)/2 + v_min
    predictions[0,2,:] = (predictions[0,2,:]+1)*(p_max - p_min)/2 + p_min

    output_train[j,:,0] = (output_train[j,:,0]+1)*(u_max - u_min)/2 + u_min
    output_train[j,:,1] = (output_train[j,:,1]+1)*(v_max - v_min)/2 + v_min
    output_train[j,:,2] = (output_train[j,:,2]+1)*(p_max - p_min)/2 + p_min

    #plot
    #plotSolution(input_train[j,:,0].cpu().numpy(), input_train[j,:,1].cpu().numpy(), predictions[0,0,:].cpu().numpy(),'u_pred_train'+str(j),'u')
    #plotSolution(input_train[j,:,0].cpu().numpy(), input_train[j,:,1].cpu().numpy(), output_train[j,:,0].cpu().numpy(),'u_truth_train'+str(j),'u')
    #plotSolution(input_train[j,:,0].cpu().numpy(), input_train[j,:,1].cpu().numpy(), np.abs(predictions[0,0,:].cpu().numpy()-output_train[j,:,0].cpu().numpy()),'u_abs_train'+str(j),'u')

    #plotSolution(input_train[j,:,0].cpu().numpy(), input_train[j,:,1].cpu().numpy(), predictions[0,1,:].cpu().numpy(),'v_pred_train'+str(j),'v')
    #plotSolution(input_train[j,:,0].cpu().numpy(), input_train[j,:,1].cpu().numpy(), output_train[j,:,1].cpu().numpy(),'v_truth_train'+str(j),'v')
    #plotSolution(input_train[j,:,0].cpu().numpy(), input_train[j,:,1].cpu().numpy(), np.abs(predictions[0,1,:].cpu().numpy()-output_train[j,:,1].cpu().numpy()),'v_abs_train'+str(j),'v')

    #plotSolution(input_train[j,:,0].cpu().numpy(), input_train[j,:,1].cpu().numpy(), predictions[0,2,:].cpu().numpy(),'p_pred_train'+str(j),'p')
    #plotSolution(input_train[j,:,0].cpu().numpy(), input_train[j,:,1].cpu().numpy(), output_train[j,:,2].cpu().numpy(),'p_truth_train'+str(j),'p')
    #plotSolution(input_train[j,:,0].cpu().numpy(), input_train[j,:,1].cpu().numpy(), np.abs(predictions[0,2,:].cpu().numpy()-output_train[j,:,2].cpu().numpy()),'p_abs_train'+str(j),'p')

    rms_u += compute_rms_error(predictions.cpu().numpy(), 0, output_train[j,:,0].cpu().numpy())
    lrms_u += compute_relative_error(predictions.cpu().numpy(), 0, output_train[j,:,0].cpu().numpy())

    rms_v += compute_rms_error(predictions.cpu().numpy(), 1, output_train[j,:,1].cpu().numpy())
    lrms_v += compute_relative_error(predictions.cpu().numpy(), 1, output_train[j,:,1].cpu().numpy())

    rms_p += compute_rms_error(predictions.cpu().numpy(), 2, output_train[j,:,2].cpu().numpy())
    lrms_p += compute_relative_error(predictions.cpu().numpy(), 2, output_train[j,:,2].cpu().numpy())

print("Average RMS of Training for u: ", rms_u / len(training_idx))
print("Average Relative of Training for u: ", lrms_u / len(training_idx))
print()
print("Average RMS of Training for v: ", rms_v / len(training_idx))
print("Average Relative of Training for v: ", lrms_v / len(training_idx))
print()
print("Average RMS of Training for p: ", rms_p / len(training_idx))
print("Average Relative of Training for p: ", lrms_p / len(training_idx))

print()
print("############################################################")
print()

###### Error analysis of the test set ######
rms_u, rms_v, rms_p = 0.0, 0.0, 0.0
lrms_u, lrms_v, lrms_p = 0.0, 0.0, 0.0

u_collection = []
v_collection = []
p_collection = []

for j in range(len(test_idx)):
    model.eval()
    input_data_pred = torch.stack((input_test[j,:,0].float(), input_test[j,:,1].float()), dim=0)
    input_data_pred = input_data_pred.unsqueeze(0)

    with torch.no_grad():
        input_data_pred = input_data_pred.to(device)
        predictions = model(input_data_pred) #shape is [1, 3, num_points]

    #back to usual x and y coordinates
    input_test[j,:,0] = (input_test[j,:,0] + 1)*(x_max - x_min)/2 + x_min
    input_test[j,:,1] = (input_test[j,:,1] + 1)*(y_max - y_min)/2 + y_min

    predictions[0,0,:] = (predictions[0,0,:]+1)*(u_max - u_min)/2 + u_min
    predictions[0,1,:] = (predictions[0,1,:]+1)*(v_max - v_min)/2 + v_min
    predictions[0,2,:] = (predictions[0,2,:]+1)*(p_max - p_min)/2 + p_min

    #plot
    plotSolution(input_test[j,:,0].cpu().numpy(), input_test[j,:,1].cpu().numpy(), predictions[0,0,:].cpu().numpy(),'u_pred_test'+str(j),r'Prediction of velocity $u$ (m/s)')
    plotSolution(input_test[j,:,0].cpu().numpy(), input_test[j,:,1].cpu().numpy(), output_test[j,:,0].cpu().numpy(),'u_truth_test'+str(j),r'Ground truth of velocity $u$ (m/s)')
    plotSolution(input_test[j,:,0].cpu().numpy(), input_test[j,:,1].cpu().numpy(), np.abs(predictions[0,0,:].cpu().numpy()-output_test[j,:,0].cpu().numpy()),'u_abs_test'+str(j),r'Absolute error of velocity $u$ (m/s)')

    plotSolution(input_test[j,:,0].cpu().numpy(), input_test[j,:,1].cpu().numpy(), predictions[0,1,:].cpu().numpy(),'v_pred_test'+str(j),r'Prediction of velocity $v$ (m/s)')
    plotSolution(input_test[j,:,0].cpu().numpy(), input_test[j,:,1].cpu().numpy(), output_test[j,:,1].cpu().numpy(),'v_truth_test'+str(j),r'Ground truth of velocity $v$ (m/s)')
    plotSolution(input_test[j,:,0].cpu().numpy(), input_test[j,:,1].cpu().numpy(), np.abs(predictions[0,1,:].cpu().numpy()-output_test[j,:,1].cpu().numpy()),'v_abs_test'+str(j),r'Absolute error of velocity $v$ (m/s)')

    plotSolution(input_test[j,:,0].cpu().numpy(), input_test[j,:,1].cpu().numpy(), predictions[0,2,:].cpu().numpy(),'p_pred_test'+str(j),r'Prediction of gauge pressure (Pa)')
    plotSolution(input_test[j,:,0].cpu().numpy(), input_test[j,:,1].cpu().numpy(), output_test[j,:,2].cpu().numpy(),'p_truth_test'+str(j),r'Ground truth of gauge pressure (Pa)')
    plotSolution(input_test[j,:,0].cpu().numpy(), input_test[j,:,1].cpu().numpy(), np.abs(predictions[0,2,:].cpu().numpy()-output_test[j,:,2].cpu().numpy()),'p_abs_test'+str(j),r'Absolute error of gauge pressure (Pa)')

    rms_u += compute_rms_error(predictions.cpu().numpy(), 0, output_test[j,:,0].cpu().numpy())
    lrms_u += compute_relative_error(predictions.cpu().numpy(), 0, output_test[j,:,0].cpu().numpy())

    rms_v += compute_rms_error(predictions.cpu().numpy(), 1, output_test[j,:,1].cpu().numpy())
    lrms_v += compute_relative_error(predictions.cpu().numpy(), 1, output_test[j,:,1].cpu().numpy())

    rms_p += compute_rms_error(predictions.cpu().numpy(), 2, output_test[j,:,2].cpu().numpy())
    lrms_p += compute_relative_error(predictions.cpu().numpy(), 2, output_test[j,:,2].cpu().numpy())

    u_collection.append(compute_relative_error(predictions.cpu().numpy(), 0, output_test[j,:,0].cpu().numpy()))
    v_collection.append(compute_relative_error(predictions.cpu().numpy(), 1, output_test[j,:,1].cpu().numpy()))
    p_collection.append(compute_relative_error(predictions.cpu().numpy(), 2, output_test[j,:,2].cpu().numpy()))

print("Average RMS of Test for u: ", rms_u / len(test_idx))
print("Average Relative of Test for u: ", lrms_u / len(test_idx))
print()
print("Average RMS of Test for v: ", rms_v / len(test_idx))
print("Average Relative of Test for v: ", lrms_v / len(test_idx))
print()
print("Average RMS of Test for p: ", rms_p / len(test_idx))
print("Average Relative of Test for p: ", lrms_p / len(test_idx))
print()
print("Maximum relative error of test for u: ", max(u_collection))
print("Index: ",u_collection.index(max(u_collection)))
print()
print("Maximum relative error of test for v: ", max(v_collection))
print("Index: ",v_collection.index(max(v_collection)))
print()
print("Maximum relative error of test for p: ", max(p_collection))
print("Index: ",p_collection.index(max(p_collection)))
print()
print("Minimum relative error of test for u: ", min(u_collection))
print("Index: ",u_collection.index(min(u_collection)))
print()
print("Minimum relative error of test for v: ", min(v_collection))
print("Index: ",v_collection.index(min(v_collection)))
print()
print("Minimum relative error of test for p: ", min(p_collection))
print("Index: ",p_collection.index(min(p_collection)))

with open("u_collection.txt", "w") as file:
    for item in u_collection:
        file.write(f"{item}\n")

with open("v_collection.txt", "w") as file:
    for item in v_collection:
        file.write(f"{item}\n")

with open("p_collection.txt", "w") as file:
    for item in p_collection:
        file.write(f"{item}\n")
