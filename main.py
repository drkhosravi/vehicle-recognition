
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import sys, os, signal
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import darknet
import time, copy, utils
from time import process_time, localtime, strftime
import vars
from train import train_model, test_model, visualize_model
import matplotlib.pyplot as plt
from torchsummary import summary
import mymodels, hogmain

def main():
    print(torch.__version__)
    # برای تقسیم داده ها به آموزش و تست به صورت خودکار
    # این تابع درصدی از داده های آموزش را ضمن حفظ اسامی پوشه ها، به پوشه تست منتقل می کند 
    if(not os.path.exists(vars.val_dir)):
        utils.create_validation_data(vars.train_dir, vars.val_dir, vars.val_split_ratio, 'jpg')


    def handler(signum, frame):
        print('Signal handler called with signal', signum)
        print('Training will finish after this epoch')
        vars.stop_training = True
        #raise OSError("Couldn't open vars.device!")

    signal.signal(signal.SIGINT, handler) # only in python version >= 3.2

    print("Start Time: ", strftime("%Y-%m-%d %H:%M:%S", localtime()))
    print("Active Mode: " + vars.mode)
    plt.ion()   # interactive mode
    ######################################################################
    # Load Data
    # Data augmentation and normalization for training
    # Just normalization for validation

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize([vars.input_size, vars.input_size]),
            #transforms.ColorJitter(0.1, 0.1, 0.1, 0.01),
            #transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize([vars.input_size, vars.input_size]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize([vars.input_size, vars.input_size]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    # image_dataset_train = {'train': datasets.ImageFolder(os.path.join(vars.data_dir, 'train'), data_transforms['train'])}
    # image_dataset_test = {'val': datasets.ImageFolder(os.path.join(vars.data_dir, 'val'), data_transforms['val'])}
    # image_dataset_train.update(image_dataset_test)
    # خط پایین با سه خط بالا برابری می کند!
    image_datasets = {x: datasets.ImageFolder(os.path.join(vars.data_dir, x), data_transforms[x])
                    for x in ['train', 'val']}

    vars.dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=vars.batch_size, shuffle=True, num_workers=0)
                for x in ['train', 'val']}

    vars.dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    vars.class_names = image_datasets['train'].classes

    # Get a batch of training data
    inputs, classes = next(iter(vars.dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    #utils.imshow(out, title=[vars.class_names[x] for x in classes])


    ######################################################################
    # Finetuning the convnet
    # Load a pretrained model and reset final fully connected layer.

    ##\\//\\//model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    if(vars.model_name.find('vgg') != -1):
        model = models.vgg11_bn(pretrained=vars.pre_trained)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, len(vars.class_names))
    elif(vars.model_name == 'resnet152'):
        model = models.resnet152(pretrained=vars.pre_trained)	
    elif(vars.model_name == 'resnet50'):
        model = models.resnet50(pretrained=vars.pre_trained)
    elif(vars.model_name == 'resnet18'):
        model = models.resnet18(pretrained=vars.pre_trained)
    elif(vars.model_name == 'googlenet'):
        model = models.googlenet(pretrained=vars.pre_trained)
    elif(vars.model_name == 'darknet53'):
        model = darknet.darknet53(1000)
        optimizer = optim.SGD(model.parameters(), lr = vars.learning_rate, momentum=0.9)
        if(vars.pre_trained):
            #model.load_state_dict( torch.load('D:\\Projects\\_Python\\car Detection\\model_best.pth.tar') )	
            checkpoint = torch.load('D:\\Projects\\_Python\\car Detection\\model_best.pth.tar')
            start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if vars.device.type == 'cuda':
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(vars.device)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
    elif(vars.model_name == 'car3conv'):
        model = mymodels.car3conv()
    elif(vars.model_name == 'car5conv'):
        model = mymodels.car5conv()	
    elif(vars.model_name == 'car2conv'):
        model = mymodels.car2conv()		
    elif(vars.model_name == 'mymodel'):
        model = mymodels.MyModel()
    elif(vars.model_name == 'mymodel2'):
        model = mymodels.MyModel2()
    else:
        return hogmain.main(None)
        

    if(vars.model_name != 'vgg11'):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(vars.class_names))

    model.to(vars.device)

    if(vars.mode == 'test'):#test
        #model.load_state_dict(torch.load("D:\\Projects\\Python\\Zeitoon Detection\"))
        model.load_state_dict(torch.load(vars.test_model))

        # log_dir = '{}-{}-{}-batch-{}'.format(vars.model_name, 'SGD', 'cuda', vars.batch_size)

        # if(vars.pre_trained):
        #	 log_dir = log_dir + '-pretrained'
        # if not os.path.exists(log_dir):
        #	 os.mkdir(log_dir)

        log_file = open(".\\Time-{}-{}.log".format(vars.model_name, vars.batch_size),"w")	
        for dev in ['cuda', 'cpu']:
            vars.device = torch.device(dev)
            model = model.to(vars.device)
            #run model on one batch to allocate required memory on device (and have more exact results)
            inputs = inputs.to(vars.device)
            outputs = model(inputs)

            s = test_model(model, vars.criterion, 'val', 100)
            log_file.write(s)
            #log_file.write('\n' + '-'*80)
        
        log_file.write(summary(model, input_size=(3, vars.input_size, vars.input_size), batch_size=-1, device=vars.device.type))
        log_file.close() 


        print(summary(model, input_size=(3, vars.input_size, vars.input_size), batch_size=-1, device=vars.device.type))
    else:
        print(summary(model, input_size=(3, vars.input_size, vars.input_size), batch_size=-1, device=vars.device.type))

        #model.load_state_dict(torch.load("C:\\Projects\\Python\\Car Detection\\darknet53-SGD-cuda-batch-32\\ep7-acc97.83-loss0.0667.pth"))

        optimizer = optim.SGD(model.parameters(), lr = vars.learning_rate, momentum=0.9)
        #optimizer = optim.Adam(model.parameters(), lr=0.05)
        # Decay LR by a factor of 0.6 every 6 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = vars.scheduler_step_size, gamma = vars.scheduler_gamma)
        model = model.to(vars.device)
        model = train_model(model, vars.criterion, optimizer, exp_lr_scheduler, vars.num_epochs)
        visualize_model(model)
        plt.ioff()
        plt.show()


if __name__ == '__main__':
	main(None)
