import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gcn import SimpleConv
from collections import OrderedDict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GradSplitter(nn.Module):
    def __init__(self, model, module_init_type , data):
        super(GradSplitter, self).__init__()
        self.model = model
        self.n_class = model.num_classes
        self.n_modules = self.n_class
        self.sign = MySign.apply
        for p in self.model.parameters():
            p.requires_grad = False
        
        self.conv_configs = [200, 8]

        self.module_params = []
        self.init_modules(module_init_type , data)

    def init_modules(self, module_init_type , data):
        for module_idx in range(self.n_modules):           
            gcn = nn.Parameter(torch.randn(self.conv_configs[0]).to(device) , requires_grad = True)
            gcn_ = nn.Parameter(torch.randn(self.conv_configs[1]).to(device) , requires_grad = True)
            setattr(self, f'module_{module_idx}_gcn1', gcn)
            setattr(self, f'module_{module_idx}_gcn2', gcn_)
            param = nn.Sequential(
                nn.Linear(self.n_class, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
            ).to(device)
            setattr(self, f'module_{module_idx}_head', param)
        print(getattr(self, f'module_{0}_head'))

    def forward(self, inputs):
        predicts = []
        for module_idx in range(self.n_modules):
            each_module_pred = self.module_predict(inputs, module_idx)
            predicts.append(each_module_pred)
        predicts = torch.cat(predicts, dim=1)
        return predicts

    def module_predict(self, x, module_idx):
        for layer_idx in [1,2]:
            gcn_layer = getattr(self.model, f'gcn{layer_idx}')
    
            x = gcn_layer(x)

            layer_param_init = getattr(self, f'module_{module_idx}_gcn{layer_idx}')
            layer_param_proc = self.sign(layer_param_init)

            x = torch.einsum('k, jk->jk', layer_param_proc, x)
        
        module_head = getattr(self, f'module_{module_idx}_head')
        head_output = torch.sigmoid(module_head(x))
        return head_output


    def get_module_params(self):
        module_used_params = []
        for module_idx in range(self.n_modules):
            each_module_params = []
            for layer_idx in [1,2]:
                # mask
                layer_param = getattr(self, f'module_{module_idx}_gcn{layer_idx}')
                each_module_params.append(self.sign(layer_param).flatten())
            module_used_params.append(torch.cat(each_module_params))
        return torch.stack(module_used_params)


class MySign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    # ctx: median
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)