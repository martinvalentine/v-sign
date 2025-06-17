import vsign.utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from vsign.modules.criterions import SeqKD
from vsign.modules import BiLSTMLayer, TemporalConv
import vsign.modules.resnet as resnet

class Identity(nn.Module): # Identity layer to replace the final fully connected layer in ResNet (return the features) 
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module): # Normalized linear layer, used 
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu')) # Initialize weights with Xavier uniform distribution

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init() # Initialize loss functions
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        #self.conv2d = getattr(models, c2d_type)(pretrained=True)
        self.conv2d = getattr(resnet, c2d_type)() # Use custom ResNet implementation
        self.conv2d.fc = Identity() # Replace the final fully connected layer with Identity to return features

        # Setup the input size for the TemporalConv layer
        self.conv1d = TemporalConv(input_size=512, # Input size is 512 for ResNet features
                                   hidden_size=hidden_size, # Hidden size for the temporal convolution
                                   conv_type=conv_type, # Type of convolution to use (e.g., '1d', '2d')
                                   use_bn=use_bn, # Use batch normalization if specified
                                   num_classes=num_classes) # Number of classes for classification
        
        # Initialize the decoder for gloss recognition
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')

        # Initialize the temporal model (BiLSTM layer)
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', 
                                          input_size=hidden_size,
                                          hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)
        
        # Set the output size of the temporal model
        if weight_norm: # Use normalized linear layer if specified
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else: # Use standard linear layer
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier
        #self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output): # Hook to handle NaN gradient to prevent issues during backpropagation
        for g in grad_input:
            g[g != g] = 0 # Replace NaN gradients with 0

    def masked_bn(self, inputs, len_x): # Masked batch normalization to handle variable-length sequences
        def pad(tensor, length): # Pad the tensor to the specified length
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        """
        Imagine you have 3 sequences in a batch with max length 10 (aka len_x[0] = 10):
            Sequence 0: 7 frames (actual length)
            Sequence 1: 5 frames (actual length)
            Sequence 2: 9 frames (actual length)

        For example: in index 1, we have a sequence of length 5
        -> start = len_x[0] * idx = 10 * 1 = 10 ()
        -> end = len_x[0] * idx + lgt = 10 * 1 + 5 = 15 (take only the actual frame, not the padding)

        Then the slice inputs[10:15] extracts only the actual frames of sequence 1, ignoring its padding. This extracted real image frames from each sign language sequence will be passed through the 2D CNN feature extractor.
        """
        x = self.conv2d(x) # Apply the 2D convolution (ResNet 18) to extract features

        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0]) 
                       for idx, lgt in enumerate(len_x)]) # Pad each sequence to the maximum length in the batch
        return x

    def forward(self, x, len_x, label=None, label_lgt=None):
        if len(x.shape) == 5:
            # videos
            batch, temp, channel, height, width = x.shape
            framewise = self.conv2d(x.permute(0,2,1,3,4)).view(batch, temp, -1).permute(0,2,1) # btc -> bct 
        else:
            # frame-wise features
            framewise = x

        # DEBUG
        # print("DEBUG: Shape of framewise before conv1d:", framewise.shape)

        conv1d_outputs = self.conv1d(framewise, len_x)

        # x: T, B, C
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']
        tm_outputs = self.temporal_model(x, lgt)
        outputs = self.classifier(tm_outputs['predictions'])
        pred = None if self.training \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        conv_pred = None if self.training \
            else self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)

        return {
            #"framewise_features": framewise,
            #"visual_features": x,
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
        }

    def criterion_calculation(self, ret_dict, label, label_lgt): # Calculate the loss based on the model outputs and labels
        loss = 0
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'SeqCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'Dist':
                loss += weight * self.loss['distillation'](ret_dict["conv_logits"], # Student model output (temporal conv)
                                                           ret_dict["sequence_logits"].detach(), # Teacher model output (BiLSTM)
                                                           use_blank=False) # Excluding the blank token
        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss
