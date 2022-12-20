# Project2
Kieran Vaudaux, Mathieu Marchand, Wanting Li

## Data augmentation
For the data augmentation, we implemented a transformer for the data with a combination of the following transformations :
- Resize image : resize the image to dimension of newHight X newWidth
- Flip : Flip the image horizontally and vertically with a probability p (for each axe)
- Rotation : Apply a rotation to the image from [-angle;+angle]
- Normalization : normalization to mean = 0, std = 1 and with max pixel value of 237 (maximum value of a pixel in the dataset)

## Neural network
The experimented neural network was a UNet. Different tests were made to determine the hyper parameters of the UNet such as the activation function, the use of bias, skipping some connections, or the activation function (not exhaustive).

Below, you will find all the configurations tested :

### Experiment 1
- Skip connection : True
- Number of epochs : 10
- number of splits : 2
- batch size : 10
- activation function : ELU with bias
- loss function : BCEWithLogitsLoss
- Optimizer : Adam
- Scheduler : torch.optim.lr_scheduler.StepLR
- Scaler :  torch.cuda.amp.GradScaler