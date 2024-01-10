import copy
import math
import matplotlib.pyplot as plt
import os
import random
import string
import sys
import time
import torch
import torchvision
from torchvision.models import MobileNet_V2_Weights

import notebooks.mototaxi_utils as moto_utils

def getDataLoaders(img_dir, hyperparams):
    batch_size = hyperparams['batch_size']
    num_samples_per_epoch = hyperparams['num_samples_per_epoch']
    num_workers = 0
    imagenet_mean = [0.485, 0.456, 0.405]
    imagenet_std = [0.229, 0.224, 0.225]

    img_transforms = {
        'train': torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(imagenet_mean, imagenet_std)
        ])
    }
    img_dataset = torchvision.datasets.ImageFolder(root=img_dir, transform=img_transforms["train"])
    #print(img_dataset)
    #print(img_dataset.class_to_idx)
    train_dataset, val_dataset, test_dataset = moto_utils.custom_random_split(img_dataset, (0.7, 0.2, 0.1),
                                                                              generator=torch.Generator().manual_seed(42)
                                                                    )
    #print(len(train_dataset.indices), train_dataset.indices)
    #print(len(val_dataset.indices), val_dataset.indices)

    train_random_sampler = torch.utils.data.RandomSampler(data_source=train_dataset,
                                                          replacement=False,
                                                          num_samples=num_samples_per_epoch,
                                                          generator=torch.Generator().manual_seed(42)
                                                          )

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               #shuffle=True,
                                               num_workers=num_workers,
                                               sampler=train_random_sampler,
                                               generator=torch.Generator().manual_seed(42)
                                               )

    val_random_sampler = torch.utils.data.RandomSampler(data_source=val_dataset,
                                                        replacement=False,
                                                        num_samples=num_samples_per_epoch,
                                                        generator=torch.Generator().manual_seed(42)
                                                        )

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             sampler=val_random_sampler,
                                             generator=torch.Generator().manual_seed(42)
                                             )
    return {'train': train_loader, 'val': val_loader}

def model_training(model, data_loaders, criterion, optimizer, scheduler, num_epochs):
    #{'train': train_loader, 'val': val_loader}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tic = time.time()
    history = {'train': {'loss': [], 'accuracy': []},
               'val': {'loss': [], 'accuracy': []}
               }
    for epoch in range(num_epochs):

        number_of_samples = {'train': 0, 'val': 0}

        for stage in ['train', 'val']:
            if stage == 'train':
                model.train()  # training mode
            else:
                model.eval()  # evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            #for i_batch, batch_inputs_y, batch_indices in enumerate(data_loaders[stage]):
            for batch_index, batch_data in enumerate(data_loaders[stage]):
                batch_inputs_y, batch_indices = batch_data
                inputs, y = batch_inputs_y  # inputs = (minibatch size, 3, 224, 224)
                number_of_samples[stage] += len(inputs)
                #print(f'Batch {batch_index}: #batch size={len(inputs)}')
                y = y.unsqueeze(1) #y, binary levels, reshape from (N, ) to (N, 1), torch.int64
                #print('inputs shape', inputs.shape)
                inputs = inputs.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(stage == 'train'):
                    yhat_logits = model(inputs)
                    #print('yhat_logits', yhat_logits) #(N, 1) logits
                    yhat = torch.where(yhat_logits < 0.5, 0, 1) #binary labels, int64.
                    #print('yhat dtype', yhat.dtype)
                    #print('yhat_logits shape', yhat_logits.shape)
                    loss = criterion(yhat_logits, y.float()) #from yhat_logits:logits, y:binary labels

                    # backward + optimize only if in training phase
                    if stage == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                #print('yhat vs y', yhat, y.data)
                running_corrects += torch.sum(yhat == y)  # comparing int64s.

            # Function criterion() does not divides by batch size, so here we do the normalization.
            epoch_loss = running_loss / number_of_samples[stage]
            epoch_accuracy = running_corrects.double() / number_of_samples[stage]

            if stage == 'train':
                scheduler.step()

            #print(f'Epoch #{epoch} -->{stage} Loss: {epoch_loss:.3f} Accuracy: {epoch_accuracy:.3f}')
            history[stage]['loss'].append(epoch_loss)
            history[stage]['accuracy'].append(epoch_accuracy)

        print(f"Epoch #{epoch:02d} -->Training Loss: {history['train']['loss'][epoch]:.3f} "
              f"Acc.: {history['train']['accuracy'][epoch]:.3f}"
              f"; Validation Loss: {history['val']['loss'][epoch]:.3f} "
              f"Acc.: {history['val']['accuracy'][epoch]:.3f}")

    time_elapsed = time.time() - tic
    print()
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    return model, history

def transfer_learning_case1(data_loaders, hyperparams):
    print("Training: Case1 (name={:}, num_samples_per_epoch={:}; batch_size={:};\n"
          "                 num_epochs={:}, learning rate: initial={:}, step_size={:}, gamma={:} )".format(
        hyperparams['name'],
        hyperparams['num_samples_per_epoch'],
        hyperparams['batch_size'],
        hyperparams['num_epochs'],
        hyperparams['lr_initial'],
        hyperparams['lr_step_size'],
        hyperparams['lr_gamma'])
    )
    num_epochs = hyperparams['num_epochs']
    lr_initial = hyperparams['lr_initial']
    lr_step_size = hyperparams['lr_step_size']
    lr_gamma = hyperparams['lr_gamma']

    model = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #torchsummary.summary(tl_model, (3, 224, 224)) #Image size=(RGB C, Height, Width)

    for param in model.parameters():
        param.requires_grad = False

    #the convolution before the last classifier: channel in:320, channels out:1280
    #after this convolution the activation tensor is (-1, 1280, 7, 7)
    #print(tl_model.classifier) #original classifier had in:1280, out:1000

    new_classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.2, inplace=False),
                                         torch.nn.Linear(in_features=model.classifier[1].in_features,
                                                         out_features=1, bias=True)
                                         )
    model.classifier = new_classifier

    model = model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), #model.classifier.parameters(),
                                 lr=lr_initial
                                 )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    model, history = model_training(model=model,
                                    data_loaders=data_loaders,
                                    criterion=criterion,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    num_epochs=num_epochs
                                    )
    return model, history

def transfer_learning_case2(data_loaders, hyperparams, starting_model):
    print("Training: Case2 (name={:}, num_samples_per_epoch={:}; batch_size={:};\n"
          "                 num_epochs={:}, learning rate: initial={:}, step_size={:}, gamma={:} )".format(
        hyperparams['name'],
        hyperparams['num_samples_per_epoch'],
        hyperparams['batch_size'],
        hyperparams['num_epochs'],
        hyperparams['lr_initial'],
        hyperparams['lr_step_size'],
        hyperparams['lr_gamma'])
    )

    num_epochs = hyperparams['num_epochs']
    lr_initial = hyperparams['lr_initial']
    lr_step_size = hyperparams['lr_step_size']
    lr_gamma = hyperparams['lr_gamma']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = copy.deepcopy(starting_model)
    layer_cutoff = 126 #138 #120 #126 #158 layers
    for i, param in enumerate(model.parameters()):
        if i < layer_cutoff:
            param.requires_grad = False
        else:
            param.requires_grad = True

    model = model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), #model.classifier.parameters(),
                                 lr=lr_initial)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    model, history = model_training(model=model,
                                    data_loaders=data_loaders,
                                    criterion=criterion,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    num_epochs=num_epochs)
    return model, history



if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Arg 'case1' or 'case2' not provided")

    case = sys.argv[1].lower()
    if case not in ['case1', 'case2']:
        sys.exit("Arg 'case1' or 'case2' not provided")

    work_dir = '../models/'
    num_workers = 0
    img_dir = '~/Downloads/dldata/mototaxi_training_images/'
    random_suffix = ''.join(random.choice(string.ascii_lowercase) for i in range(8))
    model_filename = '01_05_24_{:s}_{:s}.pth'.format(case, random_suffix)
    hyperparams = {}
    hyperparams['name'] = model_filename
    hyperparams_filename = '01_05_24_{:s}_{:s}.hparams'.format(case, random_suffix)

    if case == 'case1':
        hyperparams['batch_size'] = 32
        hyperparams['num_epochs'] = 20
        hyperparams['num_samples_per_epoch'] = 200
        hyperparams['lr_initial'] = 0.001
        hyperparams['lr_step_size'] = 1
        hyperparams['lr_gamma'] = 1.0
        data_loaders = getDataLoaders(img_dir, hyperparams)
        model, history = transfer_learning_case1(data_loaders, hyperparams)

    if case == 'case2':
        hyperparams['batch_size'] = 64
        hyperparams['num_epochs'] = 20 
        hyperparams['num_samples_per_epoch'] = 200
        hyperparams['lr_initial'] = 0.0001
        hyperparams['lr_step_size'] = 4
        hyperparams['lr_gamma'] = 1.

        data_loaders = getDataLoaders(img_dir, hyperparams)
        base_model_filename = '01_05_24_case1_sckxcduv.pth'
        base_model = torch.load(os.path.join(work_dir, base_model_filename))

        model, history = transfer_learning_case2(data_loaders, hyperparams, base_model)

    hyperparams['history'] = history
    torch.save(model, os.path.join(work_dir, model_filename))
    torch.save(hyperparams, os.path.join(work_dir, hyperparams_filename))

    os.system('mpg123 -q ~/Downloads/beep-05.mp3')
