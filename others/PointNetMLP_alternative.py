######### Library #########
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

# Check CUDA availability
print(torch.cuda.is_available())

if torch.cuda.is_available():
    device = torch.device("cuda")  # GPU available
else:
    device = torch.device("cpu")  # Only CPU available

torch.manual_seed(0)

Data = np.load('/scratch/users/kashefi/KAN_revision/FullData.npy')
data_number = Data.shape[0]
data_number = data_number - 240

print('Number of data is:')
print(data_number)

point_numbers = 1024
space_variable = 2 # 2 in 2D (x,y) and 3 in 3D (x,y,z)
cfd_variable = 3 # (u, v, p); which are the x-component of velocity, y-component of velocity, and pressure fields

input_data = zeros([data_number,point_numbers,space_variable],dtype='f')
output_data = zeros([data_number,point_numbers,cfd_variable],dtype='f')

count = 0 

for i in range(data_number+240):
    if(i<1615 or i>1854):
        input_data[count,:,0] = Data[i,:,0] # x coordinate (m)
        input_data[count,:,1] = Data[i,:,1] # y coordinate (m)
        output_data[count,:,0] = Data[i,:,3] # u (m/s)
        output_data[count,:,1] = Data[i,:,4] # v (m/s)
        output_data[count,:,2] = Data[i,:,2] # p (Pa)
        count += 1

################# Maybe need to normalize when we consider three of them #################

x_min = np.min(input_data[:,:,0])
x_max = np.max(input_data[:,:,0])
y_min = np.min(input_data[:,:,1])
y_max = np.max(input_data[:,:,1])

input_data[:,:,0] = 2*(input_data[:,:,0] - x_min)/(x_max - x_min) - 1
input_data[:,:,1] = 2*(input_data[:,:,1] - y_min)/(y_max - y_min) - 1

######## split data ########
#all_indices = np.random.permutation(data_number)
#training_idx = all_indices[:int(0.8*data_number)]
#validation_idx = all_indices[int(0.8*data_number):int(0.9*data_number)]
#test_idx = all_indices[int(0.9*data_number):]

training_idx = np.load('/scratch/users/kashefi/KAN_revision/training_idx.npy')
validation_idx = np.load('/scratch/users/kashefi/KAN_revision/validation_idx.npy')
test_idx = np.load('/scratch/users/kashefi/KAN_revision/test_idx.npy')

input_train, input_validation, input_test = input_data[training_idx,:], input_data[validation_idx,:], input_data[test_idx,:]
output_train, output_validation, output_test = output_data[training_idx,:], output_data[validation_idx,:], output_data[test_idx,:]

##### Normalize Train and Valditation Output #######
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

##### Data visualization #####

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

#number = 10 #It should be less than 'data_number'
#plot2DPointCloud(input_data[number,:,0],input_data[number,:,1],'PointCloud')
#plotSolution(input_data[number,:,0],input_data[number,:,1],output_data[number,:,0],'u_velocity','u (x-velocity component)')
#plotSolution(input_data[number,:,0],input_data[number,:,1],output_data[number,:,1],'v_velocity','v (y-velocity component)')
#plotSolution(input_data[number,:,0],input_data[number,:,1],output_data[number,:,2],'pressure','pressure')

####################################################

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

####################################################

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


############################ Define PointNet with MLP ##############################

class PointNet(nn.Module):
    def __init__(self, input_channels, output_channels, scaling=1.0):
        super(PointNet, self).__init__()

        # Shared MLP (64, 64)
        self.conv1 = nn.Conv1d(input_channels, int(64 * scaling), 1)
        self.bn1 = nn.BatchNorm1d(int(64 * scaling))
        self.conv2 = nn.Conv1d(int(64 * scaling), int(64 * scaling), 1)
        self.bn2 = nn.BatchNorm1d(int(64 * scaling))

        # Shared MLP (64, 128, 1024)
        self.conv3 = nn.Conv1d(int(64 * scaling), int(64 * scaling), 1)
        self.bn3 = nn.BatchNorm1d(int(64 * scaling))
        self.conv4 = nn.Conv1d(int(64 * scaling), int(128 * scaling), 1)
        self.bn4 = nn.BatchNorm1d(int(128 * scaling))
        self.conv5 = nn.Conv1d(int(128 * scaling), int(1024 * scaling), 1)
        self.bn5 = nn.BatchNorm1d(int(1024 * scaling))

        # Shared MLP (512, 256, 128)
        self.conv6 = nn.Conv1d(int(1024 * scaling) + int(64 * scaling), int(512 * scaling), 1)
        self.bn6 = nn.BatchNorm1d(int(512 * scaling))
        self.conv7 = nn.Conv1d(int(512 * scaling), int(256 * scaling), 1)
        self.bn7 = nn.BatchNorm1d(int(256 * scaling))
        self.conv8 = nn.Conv1d(int(256 * scaling), int(128 * scaling), 1)
        self.bn8 = nn.BatchNorm1d(int(128 * scaling))

        # Shared MLP (128, output_channels)
        self.conv9 = nn.Conv1d(int(128 * scaling), int(128 * scaling), 1)
        self.bn9 = nn.BatchNorm1d(int(128 * scaling))
        self.conv10 = nn.Conv1d(int(128 * scaling), output_channels, 1)

    def forward(self, x):

        # Shared MLP (64, 64)
        x = F.tanh(self.bn1(self.conv1(x)))
        x = F.tanh(self.bn2(self.conv2(x)))
        local_feature = x

        # Shared MLP (64, 128, 1024)
        x = F.tanh(self.bn3(self.conv3(x)))
        x = F.tanh(self.bn4(self.conv4(x)))
        x = F.tanh(self.bn5(self.conv5(x)))

        # Max pooling to get the global feature
        global_feature = F.max_pool1d(x, kernel_size=x.size(-1))
        global_feature = global_feature.expand(-1, -1, num_points)

        # Concatenate local and global features
        x = torch.cat([local_feature, global_feature], dim=1)

        # Shared MLP (512, 256, 128)
        x = F.tanh(self.bn6(self.conv6(x)))
        x = F.tanh(self.bn7(self.conv7(x)))
        x = F.tanh(self.bn8(self.conv8(x)))

        # Shared MLP (128, output_channels)
        x = F.tanh(self.bn9(self.conv9(x)))
        x = torch.tanh(self.conv10(x))
        return x

###################################################

num_samples = data_number
num_points = 1024

########################################

input_train = torch.from_numpy(input_train).float()
input_validation = torch.from_numpy(input_validation).float()
input_test = torch.from_numpy(input_test).float()

output_train = torch.from_numpy(output_train).float()
output_validation = torch.from_numpy(output_validation).float()
output_test = torch.from_numpy(output_test).float()

training_dataset = CreateDataset(input_train[:,:,0],input_train[:,:,1],output_train[:,:,0],output_train[:,:,1],output_train[:,:,2])
validation_dataset = CreateDataset(input_validation[:,:,0],input_validation[:,:,1],output_validation[:,:,0],output_validation[:,:,1],output_validation[:,:,2])

Batch_Size_Train = 128
Batch_Size_Validation = 100
dataloader_Train = DataLoader(training_dataset, batch_size=Batch_Size_Train, shuffle=True, drop_last=True)
dataloader_Validation = DataLoader(validation_dataset, batch_size=Batch_Size_Validation, shuffle=True, drop_last=True)

# Instantiate the model
input_channels = 2 #x and y
output_channels = 3 #u, v, and p
Scaling = 2.0
model = PointNet(input_channels, output_channels, scaling=Scaling)
model = model.to(device)

# Calculate the number of trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_parameters = count_parameters(model)
print(f'The number of trainable parameters in the model: {num_parameters}')

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

num_epochs = 5000

epoch_losses = []
validation_losses = []

total_training_time = 0

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

    # Validation loop
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
######################################################

def compute_rms_error(generated_cloud, variable, ground_truth):
    generated = generated_cloud[0, variable, :]
    squared_diff = (generated - ground_truth) ** 2
    mean_squared_diff = np.mean(squared_diff)
    rms_error = np.sqrt(mean_squared_diff)
    return rms_error

#######################################################

def compute_relative_error(generated_cloud, variable, ground_truth):
    generated = generated_cloud[0, variable, :]
    difference = generated - ground_truth
    norm_difference = np.linalg.norm(difference)
    norm_ground_truth = np.linalg.norm(ground_truth)
    relative_error = norm_difference / norm_ground_truth
    return relative_error

#######################################################

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

###################### Error Analysis of Training #################################
rms_u, rms_v, rms_p = 0.0, 0.0, 0.0
lrms_u, lrms_v, lrms_p = 0.0, 0.0, 0.0

for j in range(len(training_idx)):
    model.eval()
    input_data_pred = torch.stack((input_train[j,:,0].float(), input_train[j,:,1].float()), dim=0)
    input_data_pred = input_data_pred.unsqueeze(0)

    with torch.no_grad():
        input_data_pred = input_data_pred.to(device)
        predictions = model(input_data_pred) #shape is [1, 3, num_points]

    #Back to usual X and Y
    input_train[j,:,0] = (input_train[j,:,0] + 1)*(x_max - x_min)/2 + x_min
    input_train[j,:,1] = (input_train[j,:,1] + 1)*(y_max - y_min)/2 + y_min

    #Back to physical domain
    predictions[0,0,:] = (predictions[0,0,:] + 1)*(u_max - u_min)/2 + u_min
    predictions[0,1,:] = (predictions[0,1,:] + 1)*(v_max - v_min)/2 + v_min
    predictions[0,2,:] = (predictions[0,2,:] + 1)*(p_max - p_min)/2 + p_min

    output_train[j,:,0] = (output_train[j,:,0] + 1)*(u_max - u_min)/2 + u_min
    output_train[j,:,1] = (output_train[j,:,1] + 1)*(v_max - v_min)/2 + v_min
    output_train[j,:,2] = (output_train[j,:,2] + 1)*(p_max - p_min)/2 + p_min

    #Plot
    #plotSolution(input_train[j,:,0].cpu().numpy(), input_train[j,:,1].cpu().numpy(), predictions[0,0,:].cpu().numpy(),'u_pred_train'+str(j),'u')
    #plotSolution(input_train[j,:,0].cpu().numpy(), input_train[j,:,1].cpu().numpy(), output_train[j,:,0].cpu().numpy(),'u_truth_train'+str(j),'u')
    #plotSolution(input_train[j,:,0].cpu().numpy(), input_train[j,:,1].cpu().numpy(), np.abs(predictions[0,0,:].cpu().numpy()-output_train[j,:,0].cpu().numpy()),'u_abs_train'+str(j),'u')

    #plotSolution(input_train[j,:,0].cpu().numpy(), input_train[j,:,1].cpu().numpy(), predictions[0,1,:].cpu().numpy(),'v_pred_train'+str(j),'v')
    #plotSolution(input_train[j,:,0].cpu().numpy(), input_train[j,:,1].cpu().numpy(), output_train[j,:,1].cpu().numpy(),'v_truth_train'+str(j),'v')
    #plotSolution(input_train[j,:,0].cpu().numpy(), input_train[j,:,1].cpu().numpy(), np.abs(predictions[0,1,:].cpu().numpy()-output_train[j,:,1].cpu().numpy()),'v_abs_train'+str(j),'v')

    #plotSolution(input_train[j,:,0].cpu().numpy(), input_train[j,:,1].cpu().numpy(), predictions[0,2,:].cpu().numpy(),'p_pred_train'+str(j),'p')
    #plotSolution(input_train[j,:,0].cpu().numpy(), input_train[j,:,1].cpu().numpy(), output_train[j,:,2].cpu().numpy(),'p_truth_train'+str(j),'p')
    #plotSolution(input_train[j,:,0].cpu().numpy(), input_train[j,:,1].cpu().numpy(), np.abs(predictions[0,2,:].cpu().numpy()-output_train[j,:,2].cpu().numpy()),'p_abs_train'+str(j),'p')

    #Error Analysis
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
############################# Error Analysis of Test ##########################
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

    #Back to usual X and Y
    input_test[j,:,0] = (input_test[j,:,0] + 1)*(x_max - x_min)/2 + x_min
    input_test[j,:,1] = (input_test[j,:,1] + 1)*(y_max - y_min)/2 + y_min

    predictions[0,0,:] = (predictions[0,0,:] + 1)*(u_max - u_min)/2 + u_min
    predictions[0,1,:] = (predictions[0,1,:] + 1)*(v_max - v_min)/2 + v_min
    predictions[0,2,:] = (predictions[0,2,:] + 1)*(p_max - p_min)/2 + p_min

    #Plot
    plotSolution(input_test[j,:,0].cpu().numpy(), input_test[j,:,1].cpu().numpy(), predictions[0,0,:].cpu().numpy(),'u_pred_test'+str(j),r'Prediction of velocity $u$ (m/s)')
    plotSolution(input_test[j,:,0].cpu().numpy(), input_test[j,:,1].cpu().numpy(), output_test[j,:,0].cpu().numpy(),'u_truth_test'+str(j),r'Ground truth of velocity $u$ (m/s)')
    plotSolution(input_test[j,:,0].cpu().numpy(), input_test[j,:,1].cpu().numpy(), np.abs(predictions[0,0,:].cpu().numpy()-output_test[j,:,0].cpu().numpy()),'u_abs_test'+str(j),r'Absolute error of velocity $u$ (m/s)')

    plotSolution(input_test[j,:,0].cpu().numpy(), input_test[j,:,1].cpu().numpy(), predictions[0,1,:].cpu().numpy(),'v_pred_test'+str(j),r'Prediction of velocity $v$ (m/s)')
    plotSolution(input_test[j,:,0].cpu().numpy(), input_test[j,:,1].cpu().numpy(), output_test[j,:,1].cpu().numpy(),'v_truth_test'+str(j),r'Ground truth of velocity $v$ (m/s)')
    plotSolution(input_test[j,:,0].cpu().numpy(), input_test[j,:,1].cpu().numpy(), np.abs(predictions[0,1,:].cpu().numpy()-output_test[j,:,1].cpu().numpy()),'v_abs_test'+str(j),r'Absolute error of velocity $v$ (m/s)')

    plotSolution(input_test[j,:,0].cpu().numpy(), input_test[j,:,1].cpu().numpy(), predictions[0,2,:].cpu().numpy(),'p_pred_test'+str(j),r'Prediction of gauge pressure (Pa)')
    plotSolution(input_test[j,:,0].cpu().numpy(), input_test[j,:,1].cpu().numpy(), output_test[j,:,2].cpu().numpy(),'p_truth_test'+str(j),r'Ground truth of gauge pressure (Pa)')
    plotSolution(input_test[j,:,0].cpu().numpy(), input_test[j,:,1].cpu().numpy(), np.abs(predictions[0,2,:].cpu().numpy()-output_test[j,:,2].cpu().numpy()),'p_abs_test'+str(j),r'Absolute error of gauge pressure (Pa)')

    #Error Analysis
    rms_u += compute_rms_error(predictions.cpu().numpy(), 0, output_test[j,:,0].cpu().numpy())
    lrms_u += compute_relative_error(predictions.cpu().numpy(), 0, output_test[j,:,0].cpu().numpy())

    rms_v += compute_rms_error(predictions.cpu().numpy(), 1, output_test[j,:,1].cpu().numpy())
    lrms_v += compute_relative_error(predictions.cpu().numpy(), 1, output_test[j,:,1].cpu().numpy())

    rms_p += compute_rms_error(predictions.cpu().numpy(), 2, output_test[j,:,2].cpu().numpy())
    lrms_p += compute_relative_error(predictions.cpu().numpy(), 2, output_test[j,:,2].cpu().numpy())

    u_collection.append(compute_relative_error(predictions.cpu().numpy(), 0, output_test[j,:,0].cpu().numpy()))
    v_collection.append(compute_relative_error(predictions.cpu().numpy(), 1, output_test[j,:,1].cpu().numpy()))
    p_collection.append(compute_relative_error(predictions.cpu().numpy(), 2, output_test[j,:,2].cpu().numpy()))

###################################################################

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

#########################################################################

with open("u_collection.txt", "w") as file:
    for item in u_collection:
        file.write(f"{item}\n")

with open("v_collection.txt", "w") as file:
    for item in v_collection:
        file.write(f"{item}\n")

with open("p_collection.txt", "w") as file:
    for item in p_collection:
        file.write(f"{item}\n")

#########################################################################

#plot_histogram(v_collection,r'Histogram of prediction of velocity $v$')
#plot_histogram(u_collection,r'Histogram of prediction of velocity $u$')
#plot_histogram(p_collection,r'Histogram of prediction of gauge pressure')

plot_histogram(v_collection,"Av")
plot_histogram(u_collection,"Au")
plot_histogram(p_collection,"Ap")
