import torch
import torch.nn as nn


#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
pre_trained = False
learning_rate = 0.001 if pre_trained else 0.025
min_learning_rate = 0.0005
scheduler_step_size = 10 if pre_trained else 5
scheduler_gamma = 0.6
num_epochs = 25 if pre_trained else 50
stop_training = False
dataloaders = None
dataset_sizes = None
class_names = ['Bus', 'Heavy Truck', 'Medium Truck', 'Sedan', 'Pickup']
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
data_dir = 'E:/Dataset/IRVD1-256'
train_dir = data_dir + '/train/'
test_dir = data_dir + '/val/'
val_dir = data_dir + '/val/'
val_split_ratio = 0.1

# 'knn' 'svm' 'bayes' 'mlp-keras' 'mlp-torch'
# 'car3conv' 'car2conv' 'car5conv' 'mymodel' 'mymodel2' 'vgg11' 'resnet18' 'resnet50' 'resnet152' 'googlenet' 'darknet53' ''(backbone of yolov3)
model_name = 'svm'
input_size = 256 if model_name.find('vgg') != -1 else 256

mode = 'train' #train or test
#test_model = "vgg11-SGD-cuda-batch-32-99.24\\ep26-acc99.24-loss0.0346.pth"
#test_model = "darknet53-SGD-cuda-batch-32-99.38\\ep22-acc99.38-loss0.0328.pth"
#test_model = "car3conv-SGD-cuda-batch-64-95.70\\ep27-acc95.70-loss0.1420.pth"
#test_model = "car3conv-SGD-cuda-batch-64-bn-96.30\\ep48-acc96.30-loss0.1126.pth"
#test_model = "car5conv-SGD-cuda-batch-32-bn-98.63\\ep41-acc98.63-loss0.0709.pth"
test_model = "resnet152-SGD-cuda-batch-16-98.92\\ep24-acc98.92-loss0.0519.pth"
