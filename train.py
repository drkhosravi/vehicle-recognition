import torch
import torch.nn as nn
import time, copy, utils
from time import process_time, localtime, strftime
import sys, os, signal
import matplotlib.pyplot as plt
from torchsummary import summary

import vars

######################################################################
# Training the model
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    #log_dir = '{}-{}-{}-batch-{}'.format(type(model).__name__, type(optimizer).__name__, vars.device, vars.batch_size)
    log_dir = '{}-{}-{}-batch-{}'.format(vars.model_name, type(optimizer).__name__, vars.device, vars.batch_size)
    if(vars.pre_trained):
        log_dir = log_dir + '-pretrained'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_file = open(log_dir+"\\train-{}.log".format(strftime("%Y-%m-%d %H-%M", localtime())),"w")
    #log_file.write(str(model) + "\n" + '-' * 80)
    #log_file.write(summary(model, input_size=(3, vars.input_size, vars.input_size), batch_size=-1, device=vars.device.type))
    log_file.write("\nOptimizer: \n" + str(optimizer) + "\n" + '-' * 80)
    log_file.write("\nscheduler: \n" + str(scheduler) + "\n" + '-' * 80)
    log_file.flush()
    for epoch in range(vars.num_epochs):
        if(vars.stop_training):
            break
        
        log_file.write('\n' + '-' * 80)
        log_str = '\nEpoch {}/{} --'.format(epoch+1, vars.num_epochs) + strftime("%Y-%m-%d %H:%M:%S", localtime())
        log_file.write(log_str) 
        print(log_str)
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            num_batchs = vars.dataset_sizes[phase] // vars.batch_size
            # Iterate over data.
            cur_batch = 0
            for inputs, labels in vars.dataloaders[phase]:
                inputs = inputs.to(vars.device)
                labels = labels.to(vars.device)

                cur_batch += 1                

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                batch_correct = torch.sum(preds == labels.data)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += batch_correct

                if(cur_batch % 10 == 0):
                    if(phase == 'train'):
                        log_str = '\r|{:2d}/{:2d}| {:4d}/{:4d}| - {:5s} loss {:.4f} - acc {:.2f}'.format(epoch+1, vars.num_epochs, cur_batch, num_batchs, phase, loss, batch_correct.item()*100 / inputs.size(0))
                        log_file.write(log_str) 
                        log_file.flush()
                        print(log_str, end="")
                    else:
                        print('\rval batch {}/{}'.format(cur_batch, num_batchs), end="")

                # if phase == 'train':
                #     train_loss.append(loss.item())
                #     train_acc.append(batch_correct.item()*100 / inputs.size(0))
                # else:
                #     val_loss.append(loss)
                #     val_acc.append(batch_correct.item()*100 / inputs.size(0))
                # utils.plot_graphs(train_loss, val_loss, train_acc, val_acc)


            epoch_loss = running_loss / vars.dataset_sizes[phase]
            epoch_acc = running_corrects.double() / vars.dataset_sizes[phase]
            
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
                if(scheduler._last_lr[0] > vars.min_learning_rate):
                    scheduler.step()
                print("\nnew lr: {}".format(scheduler._last_lr))
                log_file.write("\nnew lr: {}".format(scheduler._last_lr)) 
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)
                utils.plot_graphs(train_loss, val_loss, train_acc, val_acc)


            log_str = '\n{:5s} Loss: {:.4f} Acc: {:.2f}'.format(phase, epoch_loss, 100*epoch_acc)
            log_file.write(log_str) 
            print(log_str, end="")
            log_file.flush()

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict()) # model.state_dict().copy()                
                torch.save(best_model_wts, '{}\\ep{}-acc{:.2f}-loss{:.4f}.pth'.format(log_dir, epoch, best_acc*100, epoch_loss))

        print()

    time_elapsed = time.time() - since

    log_str = '\nTraining complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)
    log_file.write(log_str) 
    print(log_str, end="")
            
    
    log_str = '\nBest val Acc: {:4f}'.format(best_acc)
    log_file.write(log_str) 
    print(log_str, end="")
    log_file.close() 
    utils.plot_graphs(train_loss, val_loss, train_acc, val_acc, False, log_dir)
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


######################################################################
def test_model(model, criterion, dataset='test', n_batch = 10000):
    
    log_str = '\n' + '-'*80 + '\nTest on ' + vars.device.type
    since = time.time()
     
    model.eval()   # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0

    total_test_time = 0
    # Iterate over data.
    cur_batch = 0
    total_samples = 0
    for inputs, labels in vars.dataloaders[dataset]:
        inputs = inputs.to(vars.device)
        labels = labels.to(vars.device)

        cur_batch += 1                
        if(cur_batch > n_batch):
            break

        total_samples += inputs.size(0)   
        # forward
        with torch.set_grad_enabled(False):
            t1 = time.time()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total_test_time += time.time() - t1
            loss = criterion(outputs, labels)

        # statistics
        batch_correct = torch.sum(preds == labels.data)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += batch_correct


    loss = running_loss / total_samples
    acc = running_corrects.double() / total_samples
    
    time_elapsed = time.time() - since

    log_str += '\nLoss: {:.4f} Acc: {:.2f}'.format(loss, 100*acc)

    log_str += '\nTest completed over {} samples in {:.2f}s'.format(total_samples, time_elapsed)
    log_str += '\nPrediction time per sample {:.3f}ms'.format(time_elapsed * 1000 / total_samples)

    log_str += '\n(Excluding Data Load) Test completed over {} samples in {:.2f}s'.format(total_samples, total_test_time)
    log_str += '\n(Excluding Data Load) Prediction time per sample {:.3f}ms'.format(total_test_time * 1000 / total_samples)
    
    print(log_str, end="")
    return log_str

######################################################################
# Generic function to display predictions for a few images
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(vars.dataloaders['val']):
            inputs = inputs.to(vars.device)
            labels = labels.to(vars.device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(vars.class_names[preds[j]]))
                utils.imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)




# def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
#     # Initialize these variables which will be set in this if statement. Each of these
#     #   variables is model specific.
#     model_ft = None
#     input_size = 0

#     if model_name == "resnet":
#         """ Resnet18
#         """
#         model_ft = models.resnet18(pretrained=use_pretrained)
#         set_parameter_requires_grad(model_ft, feature_extract)
#         num_ftrs = model_ft.fc.in_features
#         model_ft.fc = nn.Linear(num_ftrs, num_classes)
#         input_size = 224

#     elif model_name == "alexnet":
#         """ Alexnet
#         """
#         model_ft = models.alexnet(pretrained=use_pretrained)
#         set_parameter_requires_grad(model_ft, feature_extract)
#         num_ftrs = model_ft.classifier[6].in_features
#         model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
#         input_size = 224

#     elif model_name == "vgg":
#         """ VGG11_bn
#         """
#         model_ft = models.vgg11_bn(pretrained=use_pretrained)
#         set_parameter_requires_grad(model_ft, feature_extract)
#         num_ftrs = model_ft.classifier[6].in_features
#         model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
#         input_size = 224

#     elif model_name == "squeezenet":
#         """ Squeezenet
#         """
#         model_ft = models.squeezenet1_0(pretrained=use_pretrained)
#         set_parameter_requires_grad(model_ft, feature_extract)
#         model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
#         model_ft.num_classes = num_classes
#         input_size = 224

#     elif model_name == "densenet":
#         """ Densenet
#         """
#         model_ft = models.densenet121(pretrained=use_pretrained)
#         set_parameter_requires_grad(model_ft, feature_extract)
#         num_ftrs = model_ft.classifier.in_features
#         model_ft.classifier = nn.Linear(num_ftrs, num_classes)
#         input_size = 224

#     elif model_name == "inception":
#         """ Inception v3
#         Be careful, expects (299,299) sized images and has auxiliary output
#         """
#         model_ft = models.inception_v3(pretrained=use_pretrained)
#         set_parameter_requires_grad(model_ft, feature_extract)
#         # Handle the auxilary net
#         num_ftrs = model_ft.AuxLogits.fc.in_features
#         model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
#         # Handle the primary net
#         num_ftrs = model_ft.fc.in_features
#         model_ft.fc = nn.Linear(num_ftrs,num_classes)
#         input_size = 299

#     else:
#         print("Invalid model name, exiting...")
#         exit()

#     return model_ft, input_size

# # Initialize the model for this run
# model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# # Print the model we just instantiated
# print(model_ft)        