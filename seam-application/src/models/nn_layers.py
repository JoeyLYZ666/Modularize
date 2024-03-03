import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import copy
import re


class Binarization(autograd.Function):
    @staticmethod
    def forward(ctx, mask):
        bin_mask = (mask > 0).float()
        return bin_mask

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

class MaskLstm(nn.LSTM): 
    def __init__(self , embedding_dim , hidden_dim , is_reengineering):
        super(MaskLstm , self).__init__(embedding_dim , hidden_dim)
        self.is_reengineering = is_reengineering
        if self.is_reengineering:
            self.init_mask()
        
    def init_mask(self):
        # self.weight_mask = nn.Parameter(torch.ones(hidden_dim, hidden_dim))
        # self.bias_mask = nn.Parameter(torch.ones(hidden_dim))
        for name , param in list(self.named_parameters()): # list(a) use the copy of 'a'
            if 'weight' in name or 'bias' in name:
                mask = torch.ones_like(param.data) 
                param.requires_grad = False
                self.register_parameter(f'{name}_mask', nn.Parameter(mask, requires_grad=True)) 

    def count_weight_ratio(self):
        mask = []
        for name , param in self.named_parameters():
            if re.search('weight_.*mask' , name) or re.search('bias_.*mask' , name):
                mask.append(torch.flatten(param))
        mask = torch.cat(mask , dim=0)
        bin_mask = Binarization.apply(mask)
        weight_ratio = torch.mean(bin_mask)
        size = torch.numel(bin_mask)
        num = torch.sum(bin_mask)
        
        return num , size

    def forward(self, inputs, hidden):
        if self.is_reengineering:        
            batch_sizes = None 
            sorted_indices = None
            unsorted_indices = None

            hx = self.permute_hidden(hidden, sorted_indices) 

            self.check_forward_args(inputs, hx, batch_sizes)
            _VF = torch._C._VariableFunctions
            flat_weights_temp = []
            for name , param in self.named_parameters():
                if ('weight' in name or 'bias' in name) and hasattr(self , f'{name}_mask'):
                    bin_mask = Binarization.apply(getattr(self , f'{name}_mask'))
                    flat_weights_temp.append(param * bin_mask)
            result = _VF.lstm(inputs, hx, flat_weights_temp, self.bias, self.num_layers,  # _flat_weights represents:weights and bias
                              self.dropout, self.training, self.bidirectional, self.batch_first)
        
            output = result[0]
            hidden_result = result[1:]
        
            return output, self.permute_hidden(hidden_result, unsorted_indices) 
           
        else:
            output, hidden = super(MaskLstm, self).forward(inputs, hidden)
            return output, hidden


class MaskEmbd(nn.Embedding):
    def __init__(self , vocab_size, embedding_dim , is_reengineering):
        super(MaskEmbd , self).__init__(vocab_size , embedding_dim)
        self.is_reengineering = is_reengineering
        self.weight_mask = None
        if self.is_reengineering:
            self.init_mask()
      
    def init_mask(self):
        self.weight_mask = nn.Parameter(torch.ones(self.weight.size()))
        self.weight.requires_grad = False
        
    def count_weight_ratio(self):
        mask = []
        for name , param in self.named_parameters():
            if re.search('weight_.*mask' , name):
                mask.append(torch.flatten(param))
        mask = torch.cat(mask , dim=0)
        bin_mask = Binarization.apply(mask)
        num = torch.sum(bin_mask)
        size  = torch.numel(bin_mask)
        return num , size

    def forward(self, inputs):
        if self.is_reengineering:
            bin_weight_mask = Binarization.apply(self.weight_mask)
            weight = self.weight * bin_weight_mask
        else:
            weight = self.weight
        output = F.embedding(inputs, weight)
        return output

class MaskLinear(nn.Linear):
    def __init__(self, *args, is_reengineering, **kwargs):
        super(MaskLinear, self).__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu') # weight: out_size * in_size

        self.is_reengineering = is_reengineering
        self.weight_mask, self.bias_mask = None, None
        if self.is_reengineering:
            self.init_mask()

    def init_mask(self):
        self.weight_mask = nn.Parameter(torch.ones(self.weight.size()))
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias_mask = nn.Parameter(torch.ones(self.bias.size()))
            self.bias.requires_grad = False

    def count_weight_ratio(self):
        mask = []
        for name , param in self.named_parameters():
            if re.search('weight_.*mask' , name) or re.search('bias_.*mask' , name):
                mask.append(torch.flatten(param))
        mask = torch.cat(mask , dim=0)
        bin_mask = Binarization.apply(mask)
        num = torch.sum(bin_mask)
        size  = torch.numel(bin_mask)
        return num , size

    def forward(self, inputs):
        if self.is_reengineering:
            bin_weight_mask = Binarization.apply(self.weight_mask)
            weight = self.weight * bin_weight_mask
            if self.bias is not None:
                bin_bias_mask = Binarization.apply(self.bias_mask)
                bias = self.bias * bin_bias_mask
            else:
                bias = self.bias
        else:
            weight = self.weight
            bias = self.bias
        output = F.linear(inputs, weight, bias)
        return output