# Static augmentation

## Step to run the run.py
To run the run.py file, go to the trained_model folder and unzip the best_model3.zip file. It was too big to be pushed as a whole.

Run all the cells to generate the output masks for the prediction.

## Layout of the folders

`data` : Contains all the input data from the dataset

`experiments` : Contains the experiment files that contain the experiments made

`predict_mask` : Contains the results of the predictions

`result_exp` : Contains the results from the experiences

`submission` : Contains submission files for aicrowd

`trained_model` : Contains the models trained, stored to be used to generate masks

`utils` : Contains helper functions for the project

## Data augmentation
For the data augmentation, we implemented a transformer for the data with a combination of the following transformations :
- Resize image : resize the image to dimension of 600 X 600
- Flip : Flip the image horizontally (p = 0.5) and vertically (p = 0.1)
- Rotation : Apply a rotation to the image from [-35;+35]
- Normalization : normalization to mean = 0, std = 1 and with max pixel value of 255 

## Testing different configurations
The experimented neural network was a UNet. Different tests were made to determine the hyper parameters of the UNet such as the activation function, the use of bias, skipping some connections, or the activation function (not exhaustive).

You can find the experiment scripts in the <a href="experiments/">experiment folder</a>

Below, you will find all the configurations tested :

### Experiment 1
- Skip connection : True
- Number of epochs : 10
- number of splits : 2
- batch size : 10
- double convolution : ELU with bias
- up convolution : without dropout 
- loss function : BCEWithLogitsLoss
- Optimizer : Adam
- Scheduler : torch.optim.lr_scheduler.StepLR
- Scaler :  torch.cuda.amp.GradScaler

### Experiment 2
- Skip connection : False
- Number of epochs : 10
- number of splits : 2
- batch size : 10
- double convolution : RELU without bias
- up convolution : without dropout 
- loss function : BCEWithLogitsLoss
- Optimizer : Adam
- Scheduler : torch.optim.lr_scheduler.StepLR
- Scaler :  torch.cuda.amp.GradScaler

### Experiment 3
- Skip connection : False
- Number of epochs : 10
- number of splits : 2
- batch size : 10
- double convolution : RELU with bias
- up convolution : without dropout 
- loss function : BCEWithLogitsLoss
- Optimizer : Adam
- Scheduler : torch.optim.lr_scheduler.StepLR
- Scaler :  torch.cuda.amp.GradScaler

### Experiment 4
- Skip connection : False
- Number of epochs : 10
- number of splits : 2
- batch size : 10
- double convolution : RELU without bias
- up convolution : without dropout with bias
- loss function : BCEWithLogitsLoss
- Optimizer : Adam
- Scheduler : torch.optim.lr_scheduler.StepLR
- Scaler :  torch.cuda.amp.GradScaler

### Experiment 5
- Skip connection : True
- Number of epochs : 10
- number of splits : 2
- batch size : 10
- double convolution : ELU without bias with dropout 0.2
- up convolution : without dropout with bias
- loss function : BCEWithLogitsLoss
- Optimizer : Adam
- Scheduler : torch.optim.lr_scheduler.StepLR
- Scaler :  torch.cuda.amp.GradScaler

### Experiment 6
- Skip connection : True
- Number of epochs : 10
- number of splits : 2
- batch size : 10
- double convolution : ELU without bias with dropout 0.5
- up convolution : without dropout 
- loss function : BCEWithLogitsLoss
- Optimizer : Adam
- Scheduler : torch.optim.lr_scheduler.StepLR
- Scaler :  torch.cuda.amp.GradScaler

### Experiment 7
- Skip connection : True
- Number of epochs : 10
- number of splits : 1
- batch size : 10
- double convolution : ELU with bias with dropout TODO
- up convolution : without dropout 
- loss function : BCEWithLogitsLoss
- Optimizer : Adam
- Scheduler : torch.optim.lr_scheduler.StepLR
- Scaler :  torch.cuda.amp.GradScaler

### Experiment 8
- Skip connection : True
- Number of epochs : 10
- number of splits : 20
- batch size : 10
- double convolution : ELU with bias with dropout TODO
- up convolution : without dropout 
- loss function : BCEWithLogitsLoss
- Optimizer : Adam
- Scheduler : torch.optim.lr_scheduler.StepLR
- Scaler :  torch.cuda.amp.GradScaler

### Experiment 9
- Skip connection : True
- Number of epochs : 10
- number of splits : 2
- batch size : 10
- double convolution : ELU with bias with dropout TODO
- up convolution : without dropout 
- loss function : BCEWithLogitsLoss
- Optimizer : Adam with weight decay = 1e-04
- Scheduler : torch.optim.lr_scheduler.StepLR
- Scaler :  torch.cuda.amp.GradScaler

### Experiment 10
- Skip connection : True
- Number of epochs : 10
- number of splits : 2
- batch size : 10
- double convolution : ELU with bias with dropout TODO
- up convolution : without dropout 
- loss function : BCEWithLogitsLoss
- Optimizer : Adam with weight decay = 1e-02
- Scheduler : torch.optim.lr_scheduler.StepLR
- Scaler :  torch.cuda.amp.GradScaler