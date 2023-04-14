import matplotlib.pyplot as plt
from tqdm import tqdm
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
import torchvision
import numpy as np
from torch.utils.data import DataLoader
import torch
torch.manual_seed(17)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, kernel, skip_kernel, stride=1, bias=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=kernel[0], stride=stride, padding=kernel[1], bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel[0],
                               stride=1, padding=kernel[1], bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=skip_kernel[0], padding=skip_kernel[1], stride=stride, bias=bias),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, in_planes, num_layers, num_blocks, kernel, skip_kernel, model_name, num_classes=10, bias=True):
        if not isinstance(num_blocks, list):
            raise Exception(
                "num_blocks parameter should be a list of integer values")
        if num_layers != len(num_blocks):
            raise Exception(
                "Residual layers should be equal to the length of num_blocks list")
        super(ResNet, self).__init__()
        self.kernel = kernel
        self.skip_kernel = skip_kernel
        self.in_planes = in_planes
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=kernel[0],
                               stride=1, padding=kernel[1], bias=bias)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.num_layers = num_layers
        self.layer1 = self._make_layer(
            block, self.in_planes, num_blocks[0], stride=1, bias=bias)
        for i in range(2, num_layers+1):
            setattr(self, "layer"+str(i), self._make_layer(block, 2 *
                    self.in_planes, num_blocks[i-1], stride=2, bias=bias))
        finalshape = list(getattr(self, "layer"+str(num_layers))
                          [-1].modules())[-2].num_features
        self.multiplier = 4 if num_layers == 2 else (
            2 if num_layers == 3 else 1)
        self.linear = nn.Linear(finalshape, num_classes)
        self.path = model_name + ".pt"

    def _make_layer(self, block, planes, num_blocks, stride, bias=True):
        strides = [stride] + [1]*(num_blocks-1)
        custom_layers = []
        for stride in strides:
            custom_layers.append(
                block(self.in_planes, planes, self.kernel, self.skip_kernel, stride, bias))
            self.in_planes = planes
        return nn.Sequential(*custom_layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for i in range(1, self.num_layers+1):
            out = eval("self.layer" + str(i) + "(out)")
        out = F.avg_pool2d(out, 4*self.multiplier)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def saveToDisk(self):
        torch.save(self.state_dict(), self.path)

    def loadFromDisk(self):
        self.load_state_dict(torch.load(self.path))


class DatasetFetcher:
    def __init__(self, dataset="CIFAR10", batch_size=64):
        print("Initializing fetching %s dataset using torchvision" % (dataset))
        self.datasetObject = torchvision.datasets.__dict__.get(dataset, None)
        if self.datasetObject == None:
            raise Exception(
                "Dataset %s not available in torchvision." % (dataset))
        self.batch_size = batch_size
        self.train_transformers = []
        self.test_transformers = []
        self.workersAvailable = min(multiprocessing.cpu_count(), 14)

    def addHorizontalFlipping(self):
        self.train_transformers.append(
            torchvision.transforms.RandomHorizontalFlip())

    def addVerticalFlipping(self):
        self.train_transformers.append(
            torchvision.transforms.RandomVerticalFlip())

    def addRandomCrop(self, size=32, padding=3):
        self.train_transformers.append(
            torchvision.transforms.RandomCrop(size=size, padding=padding))

    def addHistogramEqualization(self):
        self.train_transformers.append(
            torchvision.transforms.functional.equalize)
        self.test_transformers.append(
            torchvision.transforms.functional.equalize)

    def __addToTensor(self):
        self.train_transformers.append(torchvision.transforms.ToTensor())
        self.test_transformers.append(torchvision.transforms.ToTensor())

    def __loadTrainNormalizers(self):
        params = np.load("./trainNormalizedParameters.npz")
        return params['mean'], params['std']

    def addNormalizer(self):
        self.__addToTensor()
        trainingDataset = self.datasetObject(
            root="./data", train=True, download=True)
        trainData = trainingDataset.data/255.0
        mean = trainData.mean(axis=(0, 1, 2))
        std = trainData.std(axis=(0, 1, 2))
        np.savez("./trainNormalizedParameters", mean=mean, std=std)
        self.train_transformers.append(
            torchvision.transforms.Normalize(mean=mean, std=std))
        self.test_transformers.append(
            torchvision.transforms.Normalize(mean=mean, std=std))

    def addAutoAugmentation(self):
        self.train_transformers.append(torchvision.transforms.AutoAugment(
            torchvision.transforms.AutoAugmentPolicy.CIFAR10))
        self.__addToTensor()

    def addTrivialAugmentation(self):
        self.train_transformers.append(
            torchvision.transforms.TrivialAugmentWide())
        self.__addToTensor()

    def getLoaders(self):
        if len(self.train_transformers) == 0:
            self.__addToTensor()
        trainingDataset = self.datasetObject(
            root="./data", train=True, download=True, transform=torchvision.transforms.Compose(self.train_transformers))
        testingDataset = self.datasetObject(
            root="./data", train=False, download=True, transform=torchvision.transforms.Compose(self.test_transformers))
        trainLoader = DataLoader(trainingDataset, batch_size=self.batch_size,
                                 shuffle=True, num_workers=self.workersAvailable)
        testLoader = DataLoader(testingDataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=self.workersAvailable)
        return trainLoader, testLoader

    def getTestLoader(self):
        mean, std = self.__loadTrainNormalizers()
        self.test_transformers.append(torchvision.transforms.ToTensor())
        self.test_transformers.append(
            torchvision.transforms.Normalize(mean=mean, std=std))
        testingDataset = self.datasetObject(
            root="./data", train=False, download=True, transform=torchvision.transforms.Compose(self.test_transformers))
        testLoader = DataLoader(testingDataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=self.workersAvailable)
        return testLoader


model = ResNet(BasicBlock, 32, 4, [4, 4, 4, 2], kernel=(3, 1), skip_kernel=(
    1, 0), num_classes=10, model_name='resnet', bias=True).to(device)
print("===== MODEL SUMMARY =====")
print(summary(model, input_size=(3, 32, 32)))

optimizers_dict = {
    "adam": torch.optim.Adam,
    "adagrad": torch.optim.Adagrad,
    "adadelta": torch.optim.Adadelta,
    "sgd": torch.optim.SGD
}


def main(model, data_augmentation=['trivial_aug'], epochs=100, optim="adadelta", batch_size=128, print_every=10, model_name='resnet'):
    df = DatasetFetcher(dataset="CIFAR10", batch_size=batch_size)

    for aug in data_augmentation:
        if aug == 'trivial_aug':
            df.addTrivialAugmentation()
        elif aug == 'horizontal_flip':
            df.addHorizontalFlipping()
        elif aug == 'random_crop':
            df.addRandomCrop(size=32, padding=3)
        elif aug == 'histogram_equalization':
            df.addHistogramEqualization()
        elif aug == 'normalizer':
            df.addNormalizer()
        elif aug == 'vertical_flip':
            df.addVerticalFlipping()
        elif aug == 'auto_aug':
            df.addAutoAugmentation()

    trainLoader, testLoader = df.getLoaders()

    EPOCHS = epochs
    globalBestAccuracy = 0.0
    trainingLoss = []
    testingLoss = []
    trainingAccuracy = []
    testingAccuracy = []

    # Defining Loss Function, Learning Rate, Weight Decay, Optimizer)
    lossFunction = torch.nn.CrossEntropyLoss(reduction='sum')
    learningRate = 0.1
    weightDecay = 0.0001

    optimizer_fn = optimizers_dict[optim]
    optimizer = optimizer_fn(
        model.parameters(), lr=learningRate, weight_decay=weightDecay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, EPOCHS, eta_min=learningRate/10.0)
    # print(model.eval())

    trainable_parameters = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
    print("Total Trainable Parameters : %s" % (trainable_parameters))

    if trainable_parameters > 5*(10**6):
        raise Exception("Model not under budget!")

    print(f"Total Epochs : {EPOCHS} | Optimizer : {optim} | Learning Rate : {learningRate} | Batch Size : {batch_size} | Weight Decay : {weightDecay}")
    print(f"Data Augmentation : {data_augmentation}")

    # for i in tqdm(range(EPOCHS)):
    for i in range(EPOCHS+1):
        for phase in ['train', 'test']:
            if phase == "train":
                loader = trainLoader
                model.train()
                optimizer.zero_grad()
            else:
                loader = testLoader
                model.eval()
            runningLoss = 0.0
            runningCorrects = 0
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                loss = lossFunction(output, labels)
                predicted_labels = torch.argmax(output, dim=1)
                # runningLoss += loss.item()*images.size(0)
                runningLoss += loss.item()
                runningCorrects += torch.sum(predicted_labels ==
                                             labels).float().item()
                if phase == "train":
                    loss.backward()
                    optimizer.step()
            epochLoss = runningLoss/len(loader.dataset)
            epochAccuracy = runningCorrects/len(loader.dataset)
            if phase == "train":
                scheduler.step()
                trainingLoss.append(epochLoss)
                trainingAccuracy.append(epochAccuracy)
            else:
                testingLoss.append(epochLoss)
                testingAccuracy.append(epochAccuracy)
                if epochAccuracy > globalBestAccuracy:
                    globalBestAccuracy = epochAccuracy
                    model.saveToDisk()

        if i % print_every == 0:
            print("Epoch : %s, Training Loss : %s, Testing Loss : %s, Training Accuracy : %s, Testing Accuracy : %s"
                  % (i, trainingLoss[-1], testingLoss[-1], trainingAccuracy[-1], testingAccuracy[-1]))

    print("Maximum Testing Accuracy Achieved: %s" % (max(testingAccuracy)))
    xmax = np.argmax(testingAccuracy)
    ymax = max(testingAccuracy)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    n = len(trainingLoss)
    ax1.plot(range(n), trainingLoss, '-', linewidth='3', label='Train Error')
    ax1.plot(range(n), testingLoss, '-', linewidth='3', label='Test Error')
    ax2.plot(range(n), trainingAccuracy, '-',
             linewidth='3', label='Train Accuracy')
    ax2.plot(range(n), testingAccuracy, '-',
             linewidth='3', label='Test Acuracy')
    ax2.annotate('max accuracy = %s' % (ymax), xy=(xmax, ymax), xytext=(
        xmax, ymax+0.15), arrowprops=dict(facecolor='black', shrink=0.05))
    ax1.grid(True)
    ax2.grid(True)
    ax1.legend()
    ax2.legend()
    f.savefig(model_name + "_trainTestCurve.png")


model_name = 'tra_hf_rc_ep200_optAdam_bs512'

model = ResNet(BasicBlock, 32, 4, [4, 4, 4, 2], kernel=(3, 1), skip_kernel=(
    1, 0), num_classes=10, model_name=model_name, bias=True).to(device)

main(model, data_augmentation=['trivial_aug', 'horizontal_flip', 'random_crop'],
     epochs=200, optim='adam', batch_size=512, model_name=model_name)
