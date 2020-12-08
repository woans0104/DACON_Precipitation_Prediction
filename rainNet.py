
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F



# https://discuss.pytorch.org/t/crossentropyloss-expected-object-of-type-torch-longtensor/28683/6?u=ptrblck



# https://discuss.pytorch.org/t/crossentropyloss-expected-object-of-type-torch-longtensor/28683/6?u=ptrblck

# 케라스랑 다름
# https://pytorch.org/docs/stable/nn.init.html
# https://discuss.pytorch.org/t/crossentropyloss-expected-object-of-type-torch-longtensor/28683/6
"""
def weight_init(m):
    classname = m.__class__.__name__
    print(classname)
    if isinstance(m, nn.Conv2d)  :
        print('==================')
        print(classname)
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(m.bias)
"""

def padding_size(input_, kernel_, output_, stride_):
    padding = int((output_* stride_ - stride_ - input_ + kernel_)/2)
    return padding


class rainNet(torch.nn.Module):
    def __init__(self,input_shape, output_shape, mode):

        super(rainNet, self).__init__()
        
        self.mode = mode
        kernel_size = 3
        stride = 1
        padding = padding_size(input_shape[0], kernel_size, output_shape[0], stride)
        #print(padding)
        
        # DownSampling
        self.conv1f = nn.Conv2d(9, 64, kernel_size, padding=padding)
        self.conv1f.weight = nn.init.xavier_normal_(self.conv1f.weight, gain=nn.init.calculate_gain('relu'))

        padding = padding_size(output_shape[0], kernel_size, output_shape[0], stride)
        self.conv1s = nn.Conv2d(64, 64, kernel_size, padding=padding)
        self.conv1s.weight = nn.init.xavier_normal_(self.conv1s.weight, gain=nn.init.calculate_gain('relu'))
        
        self.conv2f = nn.Conv2d(64, 128, kernel_size, padding=padding)
        self.conv2f.weight = nn.init.xavier_normal_(self.conv2f.weight, gain=nn.init.calculate_gain('relu'))
        self.conv2s = nn.Conv2d(128, 128, kernel_size, padding=padding)
        self.conv2s.weight = nn.init.xavier_normal_(self.conv2s.weight, gain=nn.init.calculate_gain('relu'))
        
        self.conv3f = nn.Conv2d(128, 256, kernel_size, padding=padding)
        self.conv3f.weight = nn.init.xavier_normal_(self.conv3f.weight, gain=nn.init.calculate_gain('relu'))

        self.conv3s = nn.Conv2d(256, 256, kernel_size, padding=padding)
        self.conv3s.weight = nn.init.xavier_normal_(self.conv3s.weight, gain=nn.init.calculate_gain('relu'))

        self.conv4f = nn.Conv2d(256, 512, kernel_size, padding=padding)
        self.conv4f.weight = nn.init.xavier_normal_(self.conv4f.weight, gain=nn.init.calculate_gain('relu'))
        self.conv4s = nn.Conv2d(512, 512, kernel_size, padding=padding)
        self.conv4s.weight = nn.init.xavier_normal_(self.conv4s.weight, gain=nn.init.calculate_gain('relu'))
        self.drop4 = nn.Dropout(0.5)
        
        self.conv5f = nn.Conv2d(512, 1024, kernel_size, padding=padding)
        self.conv5f.weight = nn.init.xavier_normal_(self.conv5f.weight, gain=nn.init.calculate_gain('relu'))
        self.conv5s = nn.Conv2d(1024, 1024, kernel_size, padding=padding)
        self.conv5s.weight = nn.init.xavier_normal_(self.conv5s.weight, gain=nn.init.calculate_gain('relu'))
        self.drop5 = nn.Dropout(0.5)
        
        # Upsampling
        self.upconv6 = nn.Upsample(size=(8,8))
        self.conv6u = nn.Conv2d(1024+512, 512, kernel_size, padding=padding)
        self.conv6u.weight = nn.init.xavier_normal_(self.conv6u.weight, gain=nn.init.calculate_gain('relu'))
        self.conv6s = nn.Conv2d(512, 512, kernel_size, padding=padding)
        self.conv6s.weight = nn.init.xavier_normal_(self.conv6s.weight, gain=nn.init.calculate_gain('relu'))
        
        self.upconv7 = nn.Upsample(size=(16,16))
        self.conv7u = nn.Conv2d(512+256, 256, kernel_size, padding=padding)
        self.conv7u.weight = nn.init.xavier_normal_(self.conv7u.weight, gain=nn.init.calculate_gain('relu'))
        self.conv7s = nn.Conv2d(256, 256, kernel_size, padding=padding)
        self.conv7s.weight = nn.init.xavier_normal_(self.conv7s.weight, gain=nn.init.calculate_gain('relu'))
        
        self.upconv8 = nn.Upsample(size=(32,32))
        self.conv8u = nn.Conv2d(256+128, 128, kernel_size, padding=padding)
        self.conv8u.weight = nn.init.xavier_normal_(self.conv8u.weight, gain=nn.init.calculate_gain('relu'))
        self.conv8s = nn.Conv2d(128, 128, kernel_size, padding=padding)
        self.conv8s.weight = nn.init.xavier_normal_(self.conv8s.weight, gain=nn.init.calculate_gain('relu'))
        
        self.upconv9 = nn.Upsample(size=(40,40))
        self.downconv9 = nn.Upsample(size=(40,40))

        self.conv9u = nn.Conv2d(128+64, 64, kernel_size, padding=padding)
        self.conv9u.weight = nn.init.xavier_normal_(self.conv9u.weight, gain=nn.init.calculate_gain('relu'))
        self.conv9s = nn.Conv2d(64, 64, kernel_size, padding=padding)
        self.conv9s.weight = nn.init.xavier_normal_(self.conv9s.weight, gain=nn.init.calculate_gain('relu'))
        self.conv9f = nn.Conv2d(64, 2, kernel_size, padding=padding)
        self.conv9f.weight = nn.init.xavier_normal_(self.conv9f.weight, gain=nn.init.calculate_gain('relu'))
        
        self.output = nn.Conv2d(2, 1, 1)

        
        
        
            
    def forward(self, x):

        conv1 = F.relu(self.conv1f(x))
        conv1 = F.relu(self.conv1s(conv1))
        pool1 = F.max_pool2d(conv1, 2)
        
        conv2 = F.relu(self.conv2f(pool1))
        conv2 = F.relu(self.conv2s(conv2))
        pool2 = F.max_pool2d(conv2, 2)

        conv3 = F.relu(self.conv3f(pool2))
        conv3 = F.relu(self.conv3s(conv3))
        pool3 = F.max_pool2d(conv3, 2)
        
        
        conv4 = F.relu(self.conv4f(pool3))
        conv4 = F.relu(self.conv4s(conv4))
        drop_conv4 = self.drop4(conv4)
        pool4 = F.max_pool2d(drop_conv4, 2)
        
        
        conv5 = F.relu(self.conv5f(pool4))
        conv5 = F.relu(self.conv5s(conv5))
        drop_conv5 = self.drop5(conv5)
        
        
        cat6 = torch.cat((conv4, self.upconv6(drop_conv5)),1)
        conv6 = F.relu(self.conv6u(cat6))
        conv6 = F.relu(self.conv6s(conv6))
        
        
        cat7 = torch.cat((conv3, self.upconv7(conv6)),1)
        conv7 = F.relu(self.conv7u(cat7))
        conv7 = F.relu(self.conv7s(conv7))
        

        cat8 = torch.cat((conv2, self.upconv8(conv7)),1)
        conv8 = F.relu(self.conv8u(cat8))
        conv8 = F.relu(self.conv8s(conv8))
        
        cat9 = torch.cat((self.downconv9(conv1), self.upconv9(conv8)),1)
        conv9 = F.relu(self.conv9u(cat9))
        conv9 = F.relu(self.conv9s(conv9))

        conv9 = self.conv9f(conv9)
        outputs = self.output(conv9)
    
        return outputs



def loss_func(y_pred, y_target):
    
    # https://stackoverflow.com/questions/54846905/pytorch-get-all-layers-of-model
    diff = y_pred - y_target
    cosh = (torch.exp(diff) + torch.exp(-diff))/2
    
    value = torch.log(cosh).view(-1)
    
    loss = torch.sum(value)/ len(value)
    
    return loss


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = rainNet(40,40, mode='regression')
    model.to(device)
