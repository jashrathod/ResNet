# ResNet

## Spring 2023 - Deep Learning - Mini Project

## Authors

Jash Rathod  
Aneek Roy

## Experiments

<!-- Experiment 1:

Model: ResNet-9
Model Parameters (in M): 3.58
Data Augmentation: AutoAugment, Horizontal Flip, Random Crop
Optimizer: AdaDelta
Epochs: 200
Batch Size: 512
Weight Decay: 0.0001
Activation: ReLU
Test Accuracy (in %): 93.84
Directory: auto_hf_rc_ep200_optAdadelta_bs512

Experiment 2:

Model: ResNet-9
Model Parameters (in M): 3.58
Data Augmentation: TrivialAugment, Horizontal Flip, Random Crop
Optimizer: Adam
Epochs: 20
Batch Size: 512
Weight Decay: 0.0001
Activation: ReLU
Test Accuracy (in %): 0.1
Directory: tra_hf_rc_ep200_optAdam_bs512

Experiment 3:

Model: ResNet-9
Model Parameters (in M): 3.58
Data Augmentation: TrivialAugment, Horizontal Flip, Random Crop
Optimizer: AdaDelta
Epochs: 200
Batch Size: 512
Weight Decay: 0.0001
Activation: ReLU
Test Accuracy (in %): 94.41
Directory: tra_hf_rc_ep200_optAdadelta_bs512

Experiment 4:

Model: ResNet-9
Model Parameters (in M): 3.58
Data Augmentation: TrivialAugment, Horizontal Flip, Random Crop
Optimizer: AdaDelta
Epochs: 200
Batch Size: 1024
Weight Decay: 0.0001
Activation: ReLU
Test Accuracy (in %): 93.78
Directory: tra_hf_rc_ep200_optAdadelta_bs1024

Experiment 5:

Model: ResNet-9
Model Parameters (in M): 3.58
Data Augmentation: TrivialAugment, Horizontal Flip, Random Crop
Optimizer: AdaDelta
Epochs: 200
Batch Size: 512
Weight Decay: 0
Activation: ReLU
Test Accuracy (in %): 94.26
Directory: tra_hf_rc_ep200_optAdadelta_bs512_wd0

Experiment 6:

Model: Modified ResNet-9
Model Parameters (in M): 4.61
Data Augmentation: TrivialAugment, Horizontal Flip, Random Crop
Optimizer: AdaDelta
Epochs: 200
Batch Size: 512
Weight Decay: 0.0001
Activation: ReLU
Test Accuracy (in %): 94.20
Directory: linear_inplane36 -->



| Model             | Model Parameters (in M) | Data Augmentation                               | Optimizer | Epochs | Batch Size | Weight Decay | Activation | Test Accuracy (in %) | Directory                                 |
| :---------------: | :---------------------: | :---------------------------------------------: | :-------: | :----: | :--------: | :---------: | :--------: | :------------------: | :---------------------------------------: |
| ResNet-9          |           3.58          | AutoAugment, Horizontal Flip, Random Crop       | AdaDelta  | 200    | 512        | 0.0001      | ReLU       | 93.84               | auto_hf_rc_ep200_optAdadelta_bs512      |
| ResNet-9          |           3.58          | TrivialAugment, Horizontal Flip, Random Crop   | Adam      | 20     | 512        | 0.0001      | ReLU       | 0.1                 | tra_hf_rc_ep200_optAdam_bs512          |
| ResNet-9          |           3.58          | TrivialAugment, Horizontal Flip, Random Crop   | AdaDelta  | 200    | 512        | 0.0001      | ReLU       | 94.41               | tra_hf_rc_ep200_optAdadelta_bs512      |
| ResNet-9          |           3.58          | TrivialAugment, Horizontal Flip, Random Crop   | AdaDelta  | 200    | 1024       | 0.0001      | ReLU       | 93.78               | tra_hf_rc_ep200_optAdadelta_bs1024     |
| ResNet-9          |           3.58          | TrivialAugment, Horizontal Flip, Random Crop   | AdaDelta  | 200    | 512        | 0           | ReLU       | 94.26               | tra_hf_rc_ep200_optAdadelta_bs512_wd0  |
| Modified ResNet-9 |           4.61          | TrivialAugment, Horizontal Flip, Random Crop   | AdaDelta  | 200    | 512        | 0.0001      | ReLU       | 94.20               | linear_inplane36                         |
