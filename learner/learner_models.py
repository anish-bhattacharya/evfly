import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM
from ConvLSTM_pytorch.convlstm import ConvLSTM
import vitfly_models

def find_output_size(model, input_size):
    mock_input = torch.rand(input_size)
    with torch.no_grad():
        mock_output = model(mock_input)
    return mock_output.shape

class InvertLayer(nn.Module):
    def forward(self, x):
        return -x

class DynamicConvNet(nn.Module):
    def __init__(self, in_channels, num_layers, kernel_sizes, kernel_strides, out_channels, activations, pool_type='max', pool_kernels=None, pool_strides=None, conv_function='conv2d', device=None, logger=None, invert_pool_input=False):
        super(DynamicConvNet, self).__init__()

        if logger is not None:
            mylogger = logger
        else:
            mylogger = print

        self.layers = nn.Sequential()

        # validate input
        assert len(kernel_sizes) == num_layers, "The length of kernel_sizes should match num_layers"
        assert len(kernel_strides) == num_layers, "The length of kernel_strides should match num_layers"
        assert len(out_channels) == num_layers, "The length of out_channels should match num_layers"
        assert len(activations) == num_layers, "The length of activations should match num_layers"
        
        if pool_kernels is None:
            pool_kernels = [2] * num_layers
        if pool_strides is None:
            pool_strides = [2] * num_layers

        # parse conv function
        if conv_function == 'conv2d':
            self.conv_function = nn.Conv2d
        elif conv_function == 'upconv2d':
            self.conv_function = nn.ConvTranspose2d
        else:
            raise NotImplementedError(f'conv_function {conv_function} not implemented. Either use conv2d or upconv2d.')

        # form network
        current_in_channels = in_channels        
        for i in range(num_layers):
            # Add convolutional layer
            self.layers.add_module(f'{conv_function}_{i}', self.conv_function(in_channels=current_in_channels,
                                                            out_channels=out_channels[i],
                                                            kernel_size=kernel_sizes[i],
                                                            stride=kernel_strides[i],
                                                            bias=False))
            
            # Add batch normalization layer
            self.layers.add_module(f'batchnorm_{i}', nn.BatchNorm2d(out_channels[i]))
            
            # Add activations layer
            if activations[i] == 'relu':
                self.layers.add_module(f'activation_{i}', nn.ReLU())
            elif activations[i] == 'sigmoid':
                self.layers.add_module(f'activation_{i}', nn.Sigmoid())
            elif activations[i] == 'tanh':
                self.layers.add_module(f'activation_{i}', nn.Tanh())
            elif activations[i] == 'leaky_relu':
                self.layers.add_module(f'activation_{i}', nn.LeakyReLU())
            elif activations[i] == 'none':
                pass
            else:
                raise NotImplementedError(f'activation {activations[i]} not implemented. Either use relu, sigmoid, tanh, or leaky_relu.')

            # Add inversion before pooling if required
            if invert_pool_input:
                self.layers.add_module(f'invert_{i}', InvertLayer())

            # Add pooling layer
            if conv_function == 'conv2d':
                if pool_type == 'max':
                    self.layers.add_module(f'pool_{i}', nn.MaxPool2d(kernel_size=pool_kernels[i], stride=pool_strides[i]))
                elif pool_type == 'avg':
                    self.layers.add_module(f'pool_{i}', nn.AvgPool2d(kernel_size=pool_kernels[i], stride=pool_strides[i]))
                elif pool_type == 'none':
                    pass
                else:
                    raise NotImplementedError(f'pool_type {pool_type} not implemented. Either use max or avg.')

            # Undo inversion after pooling if done before pooling
            if invert_pool_input:
                self.layers.add_module(f'invert_{i}', InvertLayer())

            current_in_channels = out_channels[i]

        mylogger(f'[DynamicConvNet] Initialized DynamicConvNet with in_channels={in_channels}, num_layers={num_layers}, kernel_sizes={kernel_sizes}, kernel_strides={kernel_strides}, out_channels={out_channels}, activations={activations}, pool_type={pool_type}, pool_kernels={pool_kernels}, pool_strides={pool_strides}, conv_function={conv_function}')

    def forward(self, x):
        x = self.layers(x)
        return x

class DynamicFCNet(nn.Module):
    def __init__(self, input_features, num_layers, layer_sizes, activations, dropout_p=None, device=None, logger=None):
        super(DynamicFCNet, self).__init__()

        if logger is not None:
            mylogger = logger
        else:
            mylogger = print

        self.layers = nn.Sequential()

        # validate input
        assert len(layer_sizes) == num_layers, "The length of layer_sizes should match num_layers"
        assert len(activations) == num_layers, "The length of activations should match num_layers"
        
        current_input_features = input_features
        
        for i, layer_size in enumerate(layer_sizes):
            # Add fully connected layer
            self.layers.add_module(f'fc_{i}', nn.Linear(current_input_features, layer_size))
            
            # Optionally add dropout layer
            if dropout_p is not None and dropout_p > 0:
                self.layers.add_module(f'dropout_{i}', nn.Dropout(p=dropout_p))
            
            # Add activation layer
            if activations[i] == 'relu':
                self.layers.add_module(f'activation_{i}', nn.ReLU())
            elif activations[i] == 'sigmoid':
                self.layers.add_module(f'activation_{i}', nn.Sigmoid())
            elif activations[i] == 'tanh':
                self.layers.add_module(f'activation_{i}', nn.Tanh())
            elif activations[i] == 'leaky_relu':
                self.layers.add_module(f'activation_{i}', nn.LeakyReLU())
            else:
                raise NotImplementedError(f'activation {activations[i]} not implemented. Either use relu, sigmoid, tanh, or leaky_relu.')
            
            current_input_features = layer_size

        mylogger(f'[DynamicFCNet] Initialized DynamicFCNet with input_features={input_features}, num_layers={num_layers}, layer_sizes={layer_sizes}, activations={activations}, dropout_p={dropout_p}')

    def forward(self, x):
        x = self.layers(x)
        return x

class ScaledTanh(nn.Module):
    def __init__(self, scale_factor=1.0):
        super(ScaledTanh, self).__init__()
        self.scale_factor = scale_factor
        self.tanh = nn.Tanh()

    def forward(self, x):
        scaled_tanh_output = self.scale_factor * self.tanh(x)
        return scaled_tanh_output

# ConvUNet takes in a 2-channel float voxel grid and runs a convolutional unet on it.
# input: a positive, negative polarity channels of an event frame
# output: positive, single channel depth image
class ConvUNet(nn.Module):
    def __init__(self, num_in_channels=2, num_out_channels=1, num_recurrent=0, enc_params=None, dec_params=None, input_shape=[1, 2, 60, 90], device=None, logger=None):
        super().__init__()

        if logger is not None:
            mylogger = logger
        else:
            mylogger = print

        mylogger(f'[ConvUNet] Initializing ConvUNet with num_in_channels={num_in_channels}, num_out_channels={num_out_channels}, num_recurrent={num_recurrent}')

        self.input_h, self.input_w = None, None # set in the forward function
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.num_recurrent = num_recurrent # how many recurrent layers to use

        # encoder
        if enc_params is None:
            enc_params = {
                'num_layers': 2,
                'kernel_sizes': [5, 5],
                'kernel_strides': [2, 2],
                'out_channels': [16, 64],
                'activations': ['relu', 'relu'],
                'pool_type': 'max',
                'pool_kernels': [2, 2],
                'pool_strides': [2, 2],
                'conv_function': 'conv2d'
            }
        self.enc = DynamicConvNet(in_channels=num_in_channels, num_layers=enc_params['num_layers'], kernel_sizes=enc_params['kernel_sizes'], kernel_strides=enc_params['kernel_strides'], out_channels=enc_params['out_channels'], activations=enc_params['activations'], pool_type=enc_params['pool_type'], pool_kernels=enc_params['pool_kernels'], pool_strides=enc_params['pool_strides'], conv_function=enc_params['conv_function'], logger=logger)

        # num_ch_middle = enc_params['out_channels'][-1]
        # shape_middle = (1, 2) # (2, 4)

        # figure out shape of encoder output
        # change second value of input shape to be in_channels

        input_shape = torch.Size([1, num_in_channels, input_shape[-2], input_shape[-1]])
        enc_out_shape = find_output_size(self.enc, input_shape)
        mylogger(f'[ConvUNet] Initialized encoder with input shape {input_shape} and middle shape {enc_out_shape}')
        _, self.num_ch_middle, self.shape_middle_h, self.shape_middle_w = enc_out_shape
        self.shape_middle = (self.shape_middle_h, self.shape_middle_w)
        # mylogger(enc_out_shape)
        # exit()

        # decoder
        if dec_params is None:
            dec_params = {
                'num_layers': 2,
                'kernel_sizes': [5, 5],
                'kernel_strides': [2, 2],
                'out_channels': [16, num_out_channels],
                'activations': ['relu', 'sigmoid'],
                'pool_type': 'none',
                'pool_kernels': [2, 2],
                'conv_function': 'upconv2d'
            }
        self.dec = DynamicConvNet(in_channels=self.num_ch_middle, num_layers=dec_params['num_layers'], kernel_sizes=dec_params['kernel_sizes'], kernel_strides=dec_params['kernel_strides'], out_channels=dec_params['out_channels'], activations=dec_params['activations'], pool_type=dec_params['pool_type'], pool_kernels=dec_params['pool_kernels'], pool_strides=dec_params['pool_strides'], conv_function=dec_params['conv_function'], logger=logger)

        recon_shape = find_output_size(self.dec, enc_out_shape)
        mylogger(f'[ConvUNet] Initialized decoder with input shape {enc_out_shape} and output shape {recon_shape}')

        # recurrence
        if self.num_recurrent > 0:
            lstm_size = self.num_ch_middle*self.shape_middle_h*self.shape_middle_w
            self.lstm = LSTM(input_size=lstm_size, hidden_size=lstm_size, num_layers=self.num_recurrent, dropout=0.1)
            # self.fc_lstm = nn.Linear(64, lstm_size, bias=False)

    # given evframe, form 2-channel desired input
    # input is N x 1 x 60 x 90
    # first channel negative values, second channel positive values
    def form_input(self, x):
        des_input = torch.zeros_like(x).expand(-1, 2, -1, -1)
        des_input[:, 0, :, :] = torch.where(x < 0, torch.abs(x), 0)[:, 0, :, :]
        des_input[:, 1, :, :] = torch.where(x > 0, x, 0)[:, 0, :, :]
        return des_input

    # given N x 1 x 17 x 25 output of decoder, form N x 1 x 60 x 90 output
    def form_output(self, x):
        upsampled_tensor = F.interpolate(x, size=(self.input_h, self.input_w), mode='bilinear', align_corners=False)
        return upsampled_tensor

    def down(self, x):
        # assert that the input is of shape N x 1 x 60 x 90
        # assert x.shape[1] == 1 and x.shape[2] == 60 and x.shape[3] == 90, f'input shape is {x.shape} when it should be N x 1 x 60 x 90'
        if self.num_in_channels == 2:
            x = self.form_input(x)        
        x = self.enc(x)
        return x

    def up(self, x):
        x = self.dec(x)
        x_interp = self.form_output(x)
        return x_interp, x

    # x is sequence-like with three items:
    # 0) the evframe of shape N x 1 x 60 x 90
    # 1) the desired velocity of shape N x 1
    # 2) the hidden state of shape num_layers x N x 64
    def forward(self, X):
        x = X[0]
        self.input_h, self.input_w = x.shape[2], x.shape[3]
        x = self.down(x)
        h = None
        if self.num_recurrent > 0:
            x = torch.flatten(x, 1)
            x, h = self.lstm(x, X[2])
            # x = self.fc_lstm(x)
            x = x.view(-1, self.num_ch_middle, self.shape_middle[0], self.shape_middle[1])
        x, x_upconv = self.up(x)

        return x, (x_upconv, h)

# ConvNet model that reduces an image into a 3-vector of unit norm
class VelPredictor(nn.Module):
    def __init__(self, fc_params=None, input_size=512, num_out=3, device=None, logger=None):
        super().__init__()

        if logger is not None:
            self.mylogger = logger
        else:
            self.mylogger = print

        self.input_size = input_size
        self.num_out = num_out
        self.device = device

        self.mylogger(f'[VelPredictor] Initializing VelPredictor with input_size={input_size} and num_out={num_out}')

        if fc_params is None:
            fc_params = {
                'num_layers': 3,
                'layer_sizes': [128, 32, num_out],
                'activations': ['leaky_relu', 'leaky_relu', 'tanh'],
                'dropout_p': 0.1
            }
        self.fcnet = DynamicFCNet(input_features=input_size, num_layers=fc_params['num_layers'], layer_sizes=fc_params['layer_sizes'], activations=fc_params['activations'], dropout_p=fc_params['dropout_p'], logger=logger, device=device)

    # assume we are passing in trajectories at a time, with X containing 3 items:
    # 1. images of shape [N, 1, 60, 90]
    # 2. desvel of shape [N, 1]
    # 3. tuple of initial (hidden state, cell state) each of shape [num_layers, 1, 3]
    # we are using a batch dim (dim=1) of 1
    def forward(self, X):
        x = X[0]
        x = torch.flatten(x, 1)
        x = self.fcnet(x)

        # since tanh outputs values in [-1, 1], we need to scale down elements of the output
        # by 1/sqrt(3) to be able to calculate a unit vector
        # x = x / torch.sqrt(torch.tensor(3.0, device=x.device))
        
        # if the output is of shape N x 2, then assume the output should be a 3-vector of unit norm and compute the missing first component
        if self.num_out == 2:

            radicand = torch.ones((x.shape[0], 1)).to(self.device) - torch.pow(x, 2).sum(dim=1, keepdim=True).to(self.device)
            # if radicand contains any negatives, print a warning
            if (radicand < 0).any():
                self.mylogger(f'[VelPredictor] Warning: radicand contains negatives when computing first element of 3-vector from 2-vector.')
                first_component = torch.sqrt(torch.clip(radicand, min=0.0, max=1.0))
            else:
                first_component = torch.sqrt(radicand) # avoiding clip if not needed
            x = torch.cat((first_component, x), dim=1)

        elif self.num_out == 1:

            radicand = torch.ones((x.shape[0], 1)).to(self.device) - torch.pow(x, 2).to(self.device)
            # if radicand contains any negatives, print a warning
            if (radicand < 0).any():
                self.mylogger(f'[VelPredictor] Warning: radicand contains negatives when computing first element of 2-vector from 1-vector.')
                x_component = torch.sqrt(torch.clip(radicand, min=0.0, max=1.0))
            else:
                x_component = torch.sqrt(radicand)
            z_component = torch.zeros((x.shape[0], 1)).to(self.device)
            x = torch.cat((x_component, x, z_component), dim=1)

        return x, None

class OrigUNet(nn.Module):
    def __init__(self, num_in_channels=2, num_out_channels=1, num_recurrent=0, enc_params=None, dec_params=None, input_shape=[1, 2, 260, 346], device=None, logger=None, velpred=0, fc_params=None, form_BEV=0, is_deployment=False, is_large=False, evs_min_cutoff=1e-3, skip_type='crop'):
        super().__init__()

        if logger is not None:
            mylogger = logger
        else:
            mylogger = print
        
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.num_recurrent = num_recurrent
        self.input_shape = input_shape
        self.input_h, self.input_w = input_shape[-2], input_shape[-1]
        self.velpred = velpred
        self.fc_params = fc_params
        self.enc_params = enc_params
        self.device = device
        self.form_BEV = form_BEV
        self.evs_min_cutoff = evs_min_cutoff
        self.skip_type = skip_type

        self.decoder_numch_scalar = 1 if self.skip_type == 'none' else 2

        if self.form_BEV == 1 or self.form_BEV == 2:
            self.num_in_channels = 1
        elif self.form_BEV != 0:
            raise ValueError(f'form_BEV should be 0/1/2, but is {self.form_BEV}')
        self.is_deployment = is_deployment

        mylogger(f'[OrigUNet] Initializing OrigUNet with num_in_channels={self.num_in_channels}, num_out_channels={self.num_out_channels}, num_recurrent={self.num_recurrent}, form_BEV={self.form_BEV}, is_deployment={self.is_deployment}, evs_min_cutoff={self.evs_min_cutoff}, skip_type={self.skip_type}')

        # current model is 5 layers deep (7.76M params)
        #Input: (N, 1, 260, 346)
        self.unet_e11 = nn.Conv2d(self.num_in_channels, 32, kernel_size=3, padding=0) # (N, 16, 258, 344)
        self.unet_e12 = nn.Conv2d(32, 32, kernel_size=3, padding=0) # (N, 32, 256, 342)
        self.unet_pool1 = nn.MaxPool2d(kernel_size=2, stride=2,) # (N, 32, 128, 171)

        self.unet_e21 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.unet_e22 = nn.Conv2d(64, 64, kernel_size=3, padding=0) # (N, 64, 124, 167)
        self.unet_pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # (N, 64, 62, 83)

        self.unet_e31 = nn.Conv2d(64, 128, kernel_size=3, padding=0)
        self.unet_e32 = nn.Conv2d(128, 128, kernel_size=3, padding=0) # (N, 128, 58, 79)
        self.unet_pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # (N, 128, 29, 39)

        self.unet_e41 = nn.Conv2d(128, 256, kernel_size=3, padding=0)
        self.unet_e42 = nn.Conv2d(256, 256, kernel_size=3, padding=0) # (N, 256, 25, 35)
        self.unet_pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # (N, 256, 12, 17)

        self.unet_e51 = nn.Conv2d(256, 512, kernel_size=3, padding=0)
        self.unet_e52 = nn.Conv2d(512, 512, kernel_size=3, padding=0) # (N, 512, 8, 13)
        self.unet_upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2,) # (N, 256, 16, 26)

        # size before first upconv
        self.middle_shape = (1, 512, 8, 13)
        
        # concatenate here with cropped tensor from unet_e42
        self.unet_d11 = nn.Conv2d(self.decoder_numch_scalar*256, 256, kernel_size=3, padding=0) # (N, 256, 14, 24)
        self.unet_d12 = nn.Conv2d(256, 256, kernel_size=3, padding=0) # (N, 256, 12, 22)
        self.unet_upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2,) # (N, 128, 16, 26)

        # concatenate here with cropped tensor from unet_e32
        self.unet_d21 = nn.Conv2d(self.decoder_numch_scalar*128, 128, kernel_size=3, padding=0) # (N, 128, 46, 68)
        self.unet_d22 = nn.Conv2d(128, 128, kernel_size=3, padding=0) # (N, 128, 44, 66)
        self.unet_upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2,) # (N, 64, 88, 132)

        # concatenate here with cropped tensor from unet_e22
        self.unet_d31 = nn.Conv2d(self.decoder_numch_scalar*64, 64, kernel_size=3, padding=0) # (N, 64, 86, 130)
        self.unet_d32 = nn.Conv2d(64, 64, kernel_size=3, padding=0) # (N, 64, 84, 128)
        self.unet_upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2,) # (N, 32, 168, 256)

        # concatenate here with cropped tensor from unet_e12
        self.unet_d41 = nn.Conv2d(self.decoder_numch_scalar*32, 32, kernel_size=3, padding=0) # (N, 32, 166, 254)
        self.unet_d42 = nn.Conv2d(32, 32, kernel_size=3, padding=0) # (N, 32, 164, 252)
        self.unet_out = nn.Conv2d(32, self.num_out_channels, kernel_size=1) # (N, 1, 164, 252)

        self.nonlin = nn.ReLU()

        # decoded size
        self.decoded_shape = (1, 1, 68, 148)

        if self.num_recurrent[0] > 0:
            mylogger(f'[OrigUNet] Using {self.num_recurrent[0]} recurrent layers')
            # NOTE that while PyTorch LSTM can take (L, H_in) unbatched input, ConvLSTM requires a batch dimension that can be first when batch_first=True
            self.lstm = ConvLSTM(input_dim=self.middle_shape[1], hidden_dim=[self.middle_shape[1]]*self.num_recurrent[0], num_layers=self.num_recurrent[0], kernel_size=(1, 1), bias=False, batch_first=True, return_all_layers=False)

        ## Velocity Predictor

        if self.velpred > 0:

            if self.velpred == 1:

                mylogger(f'[OrigUNet] self.velpred == 1; Using velocity predictor with a ConvNet encoder and FC head.')
                # :\nenc_params {enc_params}\nfc_params {fc_params}; input_shape_enc = {input_shape} ')

                self.convnet_velpred = DynamicConvNet(in_channels=1, num_layers=enc_params['num_layers'], kernel_sizes=enc_params['kernel_sizes'], kernel_strides=enc_params['kernel_strides'], out_channels=enc_params['out_channels'], activations=enc_params['activations'], pool_type=enc_params['pool_type'], pool_kernels=enc_params['pool_kernels'], pool_strides=enc_params['pool_strides'], conv_function=enc_params['conv_function'], invert_pool_input=enc_params['invert_pool_inputs'], logger=mylogger, device=device)
                
                mylogger(f'[OrigUNet] Input size to velpred: {[1, 1, input_shape[-2], input_shape[-1]]}')
                input_shape_enc = torch.Size([1, 1, input_shape[-2], input_shape[-1]])

            elif self.velpred == 11:
                    
                mylogger(f'[OrigUNet] self.velpred == 11; Using velocity predictor with a ConvNet encoder and FC head.')
                # :\nenc_params {enc_params}\nfc_params {fc_params}; input_shape_enc = {self.decoded_shape}')

                self.convnet_velpred = DynamicConvNet(in_channels=self.decoded_shape[1], num_layers=enc_params['num_layers'], kernel_sizes=enc_params['kernel_sizes'], kernel_strides=enc_params['kernel_strides'], out_channels=enc_params['out_channels'], activations=enc_params['activations'], pool_type=enc_params['pool_type'], pool_kernels=enc_params['pool_kernels'], pool_strides=enc_params['pool_strides'], conv_function=enc_params['conv_function'], invert_pool_input=enc_params['invert_pool_inputs'], logger=mylogger, device=device)
                
                mylogger(f'[OrigUNet] Input size to velpred: {[1, self.decoded_shape[1], self.decoded_shape[2], self.decoded_shape[3]]}')
                input_shape_enc = torch.Size([1, self.decoded_shape[1], self.decoded_shape[2], self.decoded_shape[3]])

            elif self.velpred == 2:

                mylogger(f'[OrigUNet] self.velpred == 2; Using velocity predictor with a ConvNet encoder and ConvNet head.')
                # :\nenc_params {enc_params}\nfc_params {fc_params}; input_shape_enc = {self.middle_shape}')

                self.convnet_velpred = DynamicConvNet(in_channels=self.middle_shape[1], num_layers=enc_params['num_layers'], kernel_sizes=enc_params['kernel_sizes'], kernel_strides=enc_params['kernel_strides'], out_channels=enc_params['out_channels'], activations=enc_params['activations'], pool_type=enc_params['pool_type'], pool_kernels=enc_params['pool_kernels'], pool_strides=enc_params['pool_strides'], conv_function=enc_params['conv_function'], invert_pool_input=enc_params['invert_pool_inputs'], logger=mylogger, device=device)
                
                mylogger(f'[OrigUNet] Input size to velpred: {[1, self.middle_shape[1], self.middle_shape[2], self.middle_shape[3]]}')
                input_shape_enc = torch.Size([1, self.middle_shape[1], self.middle_shape[2], self.middle_shape[3]])

            self.convnet_velpred_outsize = find_output_size(self.convnet_velpred, input_shape_enc)

            mylogger(f'[OrigUNet] Calculated self.convnet_velpred_outsize = {self.convnet_velpred_outsize}')

            if self.num_recurrent[1] > 0:
                self.lstm_velpred = LSTM(input_size=self.convnet_velpred_outsize[1]*self.convnet_velpred_outsize[2]*self.convnet_velpred_outsize[3], hidden_size=self.convnet_velpred_outsize[1]*self.convnet_velpred_outsize[2]*self.convnet_velpred_outsize[3], num_layers=self.num_recurrent[1], dropout=0.1)
                mylogger(f'[OrigUNet] LSTM for velocity prediction has {sum(p.numel() for p in self.lstm_velpred.parameters() if p.requires_grad):,} parameters.')

            self.velpred_head = VelPredictor(fc_params=fc_params, input_size=self.convnet_velpred_outsize[1]*self.convnet_velpred_outsize[2]*self.convnet_velpred_outsize[3], num_out=1, device=device, logger=mylogger)

            # print number of parameters for velpred parts
            mylogger(f'[OrigUNet] ConvNet for velocity prediction has {sum(p.numel() for p in self.convnet_velpred.parameters() if p.requires_grad):,} parameters.')
            mylogger(f'[OrigUNet] FCNet for velocity prediction has {sum(p.numel() for p in self.velpred_head.fcnet.parameters() if p.requires_grad):,} parameters.')

    # given evframe, form 2-channel desired input
    # first channel negative values, second channel positive values
    def form_input(self, x):
        x[x.abs()<self.evs_min_cutoff] = 0.0
        if self.form_BEV == 0:
            des_input = torch.zeros_like(x).expand(-1, 2, -1, -1)
            des_input[:, 0, :, :] = torch.where(x < 0, torch.abs(x), torch.tensor(0.0).float().to(self.device))[:, 0, :, :]
            des_input[:, 1, :, :] = torch.where(x > 0, x, torch.tensor(0.0).float().to(self.device))[:, 0, :, :]
        
        # single-channel absolute value of evframe
        elif self.form_BEV == 1:
            des_input = torch.abs(x)
        
        # single-channel binary event mask
        elif self.form_BEV == 2:
            des_input = torch.zeros_like(x)
            des_input[x!=0.0] = 1.0

        else:
            raise ValueError(f'form_BEV should be 0/1/2, but is {self.form_BEV}')
        return des_input

    def form_output(self, x):
        upsampled_tensor = F.interpolate(x, size=(self.input_h, self.input_w), mode='bilinear', align_corners=False)
        upconv_tensor = x

        # if outputting the evframe in 2 channels, form back into a single-channel evframe before outputting
        if self.num_out_channels == 2:
            upsampled_tensor = upsampled_tensor[:, 1, :, :] - upsampled_tensor[:, 0, :, :]
            upsampled_tensor.unsqueeze_(1)

            upconv_tensor = x[:, 1, :, :] - x[:, 0, :, :]
            upconv_tensor.unsqueeze_(1)

        return upsampled_tensor, upconv_tensor

    def skip(self, y, big, small):
        if self.skip_type == 'crop':
            skip_out = y[:, :, big[0]//2-small[0]//2 : big[0]//2+small[0]//2, big[1]//2-small[1]//2 : big[1]//2+small[1]//2 ]
        elif self.skip_type == 'interp':
            skip_out = F.interpolate(y, size=(small[0], small[1]), mode='bilinear', align_corners=False)
        elif self.skip_type == 'none':
            skip_out = None
        else:
            raise ValueError(f'[LEARNER_MODELS/ORIGUNET] skip_type should be crop/interp/none, but is {self.skip_type}.')
        return skip_out

    def forward(self, x):
        im = x[0]
        if self.num_in_channels == 2 or self.form_BEV > 0:
            im = self.form_input(im)
        
        if x[2] is None:
            x[2] = (None, None)

        # st_unet = time.time()

        # encoder

        y_e1 = self.nonlin(self.unet_e12(self.nonlin(self.unet_e11(im))))
        unet_enc1 = self.unet_pool1(y_e1)
        y_e2 = self.nonlin(self.unet_e22(self.nonlin(self.unet_e21(unet_enc1))))
        unet_enc2 = self.unet_pool2(y_e2)
        y_e3 = self.nonlin(self.unet_e32(self.nonlin(self.unet_e31(unet_enc2))))
        unet_enc3 = self.unet_pool3(y_e3)
        y_e4 = self.nonlin(self.unet_e42(self.nonlin(self.unet_e41(unet_enc3))))
        unet_enc4 = self.unet_pool4(y_e4)
        y_e5 = self.nonlin(self.unet_e52(self.nonlin(self.unet_e51(unet_enc4))))

        h_unet = None
        if self.num_recurrent[0] > 0:
            y_e5_lstm, h_unet = self.lstm(y_e5.unsqueeze(0), x[2][0])
            y_e5 = y_e5_lstm[0].squeeze(0)

        y_upconv = None # torch.zeros((), device=self.device)
        y_interp = None # torch.zeros((), device=self.device)

        # decoder

        if not self.is_deployment or (self.is_deployment and (self.velpred == 1 or self.velpred == 11)):

            # big = (25, 35)
            # small = (16, 26)
            # cropped_enc = y_e4[:, :, big[0]//2-small[0]//2 : big[0]//2+small[0]//2, big[1]//2-small[1]//2 : big[1]//2+small[1]//2 ]
            skipped_enc = self.skip(y_e4, (25, 35), (16, 26))
            concat_input = torch.cat((skipped_enc, self.unet_upconv1(y_e5)), 1) if skipped_enc is not None else self.unet_upconv1(y_e5)
            y_d1 = self.nonlin(self.unet_d12(self.nonlin(self.unet_d11( concat_input ))))

            # big = (58, 79)
            # small = (24, 44)
            # cropped_enc = y_e3[:, :, big[0]//2-small[0]//2 : big[0]//2+small[0]//2, big[1]//2-small[1]//2 : big[1]//2+small[1]//2 ]
            skipped_enc = self.skip(y_e3, (58, 79), (24, 44))
            concat_input = torch.cat((skipped_enc, self.unet_upconv2(y_d1)), 1) if skipped_enc is not None else self.unet_upconv2(y_d1)
            y_d2 = self.nonlin(self.unet_d22(self.nonlin(self.unet_d21( concat_input ))))

            # big = (124, 167)
            # small = (40, 80)
            # cropped_enc = y_e2[:, :, big[0]//2-small[0]//2 : big[0]//2+small[0]//2, big[1]//2-small[1]//2 : big[1]//2+small[1]//2 ]
            skipped_enc = self.skip(y_e2, (124, 167), (40, 80))
            concat_input = torch.cat((skipped_enc, self.unet_upconv3(y_d2)), 1) if skipped_enc is not None else self.unet_upconv3(y_d2)
            y_d3 = self.nonlin(self.unet_d32(self.nonlin(self.unet_d31( concat_input ))))

            # big = (256, 342)
            # small = (72, 152)
            # cropped_enc = y_e1[:, :, big[0]//2-small[0]//2 : big[0]//2+small[0]//2, big[1]//2-small[1]//2 : big[1]//2+small[1]//2 ]
            skipped_enc = self.skip(y_e1, (256, 342), (72, 152))
            concat_input = torch.cat((skipped_enc, self.unet_upconv4(y_d3)), 1) if skipped_enc is not None else self.unet_upconv4(y_d3)
            y_d4 = self.nonlin(self.unet_d42(self.nonlin(self.unet_d41( concat_input ))))
            
            y_upconv = self.unet_out(y_d4)

            y_interp, y_upconv = self.form_output(y_upconv)

        # velocity prediction

        # make tensor [1, 0, 0] repeat to first dim length of batch
        y_vel = torch.Tensor([1., 0., 0.]) # default value is forward full speed
        y_vel = y_vel.repeat(x[0].shape[0], 1)
        
        h_velpred = None
        if self.velpred > 0:

            if self.velpred == 1:

                y_postconvnet_velpred = self.convnet_velpred(y_interp)

            elif self.velpred == 11:

                y_postconvnet_velpred = self.convnet_velpred(y_upconv)
            
            elif self.velpred == 2:
            
                y_postconvnet_velpred = self.convnet_velpred(y_e5)
            
            y_postconvnet_velpred = torch.flatten(y_postconvnet_velpred, 1)
            
            if self.num_recurrent[1] > 0:
            
                y_postconvnet_velpred, h_velpred = self.lstm_velpred(y_postconvnet_velpred, x[2][1])
            
            y_vel, _ = self.velpred_head([y_postconvnet_velpred])

        return y_vel, (y_interp, y_upconv, (h_unet, h_velpred))

class OrigUNet_w_VITFLY_ViTLSTM(nn.Module):
    def __init__(self, num_in_channels=2, num_out_channels=1, num_recurrent=0, enc_params=None, dec_params=None, input_shape=[1, 2, 260, 346], device=None, logger=None, old_model=False, velpred=False, fc_params=None, form_BEV=0, is_deployment=False, evs_min_cutoff=1e-3, skip_type='crop'):
        super().__init__()
        # evs -> depth
        self.origunet = OrigUNet(num_in_channels=num_in_channels, num_out_channels=num_out_channels, num_recurrent=num_recurrent, enc_params=enc_params, dec_params=dec_params, input_shape=input_shape, device=device, logger=logger, velpred=velpred, fc_params=fc_params, form_BEV=form_BEV, is_deployment=is_deployment, evs_min_cutoff=evs_min_cutoff, skip_type=skip_type)
        # depth -> vel
        self.vitfly_vitlstm = vitfly_models.LSTMNetVIT()

        # print number of parameters
        print(f'[OrigUNet_w_VITFLY_ViTLSTM] Number of parameters: {sum(p.numel() for p in self.parameters()):,}')

    def forward(self, X):

        x = X[0]
        # st_unet = time.time()
        _, (x_depth, y_upconv, (h_unet, h_velpred)) = self.origunet([x, None, X[2]])
        x_depth_input = torch.clip(x_depth * 2, 0.0, 1.0) # * 2 scaling is needed to roughly match the depth scales that VITFLY_ViTLSTM was trained on
        x_vel, h_vitlstm = self.vitfly_vitlstm([x_depth_input, X[1], None, X[3]])
        return x_vel, (x_depth, y_upconv, ((h_unet, h_velpred), h_vitlstm))
