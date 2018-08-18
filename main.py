from __future__ import print_function


from loader import image_loader, imshow
from nn_style_transfer import run_style_transfer

import argparse
import torch
import torchvision.models as models
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Perform style transfer using CNNs and pytorch')
parser.add_argument('-s', '--style', required=True, help='The style image')
parser.add_argument('-c', '--content', required=True, help='The content image')
args = parser.parse_args()

style_img = image_loader(args.style)
content_img = image_loader(args.content)

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])


assert style_img.size() == content_img.size()


cnn = models.vgg19(pretrained=True).features.eval()

input_img = content_img.clone()
# if you want to use a white noise instead uncomment the below line:
# input_img = torch.randn(content_img.data.size(), device=device)

# add the original input image to the figure:
plt.figure()
imshow(input_img, title='Input Image')

output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

plt.figure()
imshow(output, title='Output Image')

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()
