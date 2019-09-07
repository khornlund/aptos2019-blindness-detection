import PIL
import math
import cv2
import torch
from torch.autograd import Variable
from torch import nn
from torchvision import transforms
import numpy
import numpy as np
from collections import OrderedDict
from PIL import Image, ImageFile
import os
from os.path import join
from tqdm import tqdm
import pandas as pd
from os import listdir
import torch.utils.model_zoo as model_zoo


# -------
# Setup
# -------

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

ImageFile.LOAD_TRUNCATED_IMAGES = True

data_dir = '../input/aptos2019-blindness-detection/'
train_dir = data_dir + '/train_images/'
test_dir = data_dir + '/test_images/'
nThreads = 4
batch_size = 32
use_gpu = torch.cuda.is_available()

# ----------
# Classes
# ----------


class GenericDataset():
    def __init__(self, labels, root_dir, subset=False, transform=None):
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx, 0]  # file name
        fullname = join(self.root_dir, img_name)
        image = Image.open(fullname).convert('RGB')
        labels = self.labels.iloc[idx, 2]  # category_id
        #         print (labels)
        if self.transform:
            image = self.transform(image)
        return image, labels

    @staticmethod
    def find_classes(fullDir):
        classes = [d for d in os.listdir(fullDir) if os.path.isdir(os.path.join(fullDir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        #     idx_to_class = dict(zip(range(len(classes)), classes))

        train = []
        for index, label in enumerate(classes):
            path = fullDir + label + '/'
            for file in listdir(path):
                train.append(['{}/{}'.format(label, file), label, index])

        df = pd.DataFrame(train, columns=['file', 'category', 'category_id', ])

        return classes, class_to_idx, idx_to_class, df

    @staticmethod
    def find_classes_retino(fullDir):

        df_labels = pd.read_csv("../input/aptos2019-blindness-detection/train.csv", sep=',')
        class_to_idx = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}

        idx_to_class = dict(zip(class_to_idx.values(), class_to_idx.keys()))
        classes = ['0', '1', '2', '3', '4']

        print('Sorted Classes: {}'.format(classes))
        print('class_to_idx: {}'.format(class_to_idx))
        print('num_to_class: {}'.format(idx_to_class))

        train = []
        for index, row in (df_labels.iterrows()):
            id = (row['id_code'])
            # currImage = os.path.join(fullDir, num_to_class[(int(row['melanoma']))] + '/' + id + '.jpg')
            currImage_on_disk = os.path.join(fullDir, id + '.png')
            if os.path.isfile(currImage_on_disk):
                if (int(row['diagnosis'])) == 0:
                    trait = '0'
                elif (int(row['diagnosis'])) == 1:
                    trait = '1'
                elif (int(row['diagnosis'])) == 2:
                    trait = '2'
                elif (int(row['diagnosis'])) == 3:
                    trait = '3'
                elif (int(row['diagnosis'])) == 4:
                    trait = '4'

                train.append(['{}'.format(currImage_on_disk), trait, class_to_idx[trait]])
        df = pd.DataFrame(train, columns=['file', 'category', 'category_id', ])
        if os.path.isfile('full_retino_labels.csv'):
            os.remove('full_retino_labels.csv')
        df.to_csv('full_retino_labels.csv', index=None)
        return classes, class_to_idx, idx_to_class, df


classes, class_to_idx, idx_to_class, df = GenericDataset.find_classes_retino(train_dir)


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def initialize_pretrained_model(model, num_classes, settings):
    assert num_classes == settings['num_classes'], \
        'num_classes should be {}, but is {}'.format(
            settings['num_classes'], num_classes)
    model.load_state_dict(model_zoo.load_url(settings['url']))
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']


def se_resnext50_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    return model


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def get_head(nf: int, n_classes):
    model = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        Flatten(),
        nn.Linear(nf, n_classes)
    )
    return model


def init_weights(model):
    for i, module in enumerate(model):
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if module.weight is not None:
                nn.init.uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            if getattr(module, "weight_v", None) is not None:
                print("Initing linear with weight normalization")
                assert model[i].weight_g is not None
            else:
                nn.init.kaiming_normal_(module.weight)
                print("Initing linear")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    return model


def get_seresnet_model(n_classes: int = 10, pretrained: bool = False):
    full = se_resnext50_32x4d(
        pretrained='imagenet' if pretrained else None)
    model = nn.Sequential(
        nn.Sequential(full.layer0, full.layer1, full.layer2, full.layer3[:3]),
        nn.Sequential(full.layer3[3:], full.layer4),
        get_head(2048, n_classes))
    print(" | ".join([
        "{:,d}".format(np.sum([p.numel() for p in x.parameters()])) for x in model]))
    if pretrained:
        init_weights(model[-1])
        return model
    return init_weights(model)


# -----------
# Execution
# -----------


def crop_image1(img, tol=7):
    # img is image data
    # tol  is tolerance

    mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = get_seresnet_model(n_classes=len(classes), pretrained=False)
    model = torch.nn.DataParallel(model, device_ids=list(range(1)))
    model = model.cuda()
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model, checkpoint['class_to_idx'], checkpoint['idx_to_class']


# Get index to class mapping
loaded_model1, class_to_idx, idx_to_class = load_checkpoint(
    '../input/pth-best/seresnext50_MixUpSoftmaxLoss_0.8571428571428571')
loaded_model2, class_to_idx, idx_to_class = load_checkpoint(
    '../input/pth-best/seresnext50_MixUpSoftmaxLoss_0.8454545454545455')
loaded_model3, class_to_idx, idx_to_class = load_checkpoint(
    '../input/pth-best/seresnext50_MixUpSoftmaxLoss_0.8487394957983193')

mdls = [loaded_model1, loaded_model2, loaded_model3]

AUG_IMG_SIZE = 512


def david(image_path):
    image = Image.open(image_path).convert('RGB')
    image = cv2.cvtColor(numpy.array(image), cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (AUG_IMG_SIZE, AUG_IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 30), -4, 128)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return image


def nodavid(image_path):
    image = Image.open(image_path).convert('RGB')
    return image


# pil
tensor_transform_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_transform_tsr = transforms.Compose([
        transforms.Resize((AUG_IMG_SIZE, AUG_IMG_SIZE), interpolation=Image.NEAREST),
        transforms.CenterCrop(AUG_IMG_SIZE),
        tensor_transform_norm
    ])

train_transform_tsr = transforms.Compose([
    #         transforms.ToPILImage(),
    transforms.Resize((AUG_IMG_SIZE, AUG_IMG_SIZE), interpolation=Image.NEAREST),
    transforms.CenterCrop(AUG_IMG_SIZE),
    transforms.RandomApply([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20, resample=PIL.Image.BICUBIC),
        transforms.RandomAffine(0, translate=(
            0.2, 0.2), resample=PIL.Image.BICUBIC),
        transforms.RandomAffine(0, shear=20, resample=PIL.Image.BICUBIC),
        transforms.RandomAffine(0, scale=(0.8, 1.2),
                                resample=PIL.Image.BICUBIC)
    ], p=0.85),
    tensor_transform_norm,
])


def predictOne(image_path, topk=1, num_tta=4):
    preds = []

    image = david(image_path)
    image_orig = image
    image = test_transform_tsr(image_orig)
    image = image.unsqueeze(0)
    image.cuda()

    for m in mdls:
        preds.append(m.forward(Variable(image)).data.cpu().numpy()[0])

    for tta in range(num_tta):
        image = train_transform_tsr(image_orig)
        image = image.unsqueeze(0)
        image.cuda()
        for m in mdls:
            preds.append(m.forward(Variable(image)).data.cpu().numpy()[0])

    preds = numpy.mean(numpy.array(preds), axis=0)
    pobabilities = numpy.exp(preds)
    top_idx = np.argsort(pobabilities)[-topk:][::-1]
    top_class = [idx_to_class[x] for x in top_idx]
    top_probability = pobabilities[top_idx]
    return top_class[0]


def testModel(test_dir):
    for m in mdls:
        m.cuda()
        m.eval()

    topk = 1

    columns = ['id_code', 'diagnosis']
    df_pred = pd.DataFrame(data=np.zeros((0, len(columns))), columns=columns)

    for index, row in tqdm(sample_submission.iterrows(), total=sample_submission.shape[0]):
        currImage = os.path.join(test_dir, row['id_code'])
        currImage = currImage + ".png"
        if os.path.isfile(currImage):
            with torch.no_grad():
                df_pred = df_pred.append(
                    {'id_code': row['id_code'], 'diagnosis': predictOne(currImage)}, ignore_index=True)
    return df_pred


sample_submission = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
sample_submission.columns = ['id_code', 'diagnosis']
sample_submission.head(3)

df_pred = testModel(test_dir)
df_pred.to_csv("submission.csv", columns=('id_code', 'diagnosis'), index=None)
