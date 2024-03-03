import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.nn_layers import MaskLinear , MaskEmbd , MaskLstm , Binarization

class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size,
                  label_size, batch_size, use_gpu,
                    is_reengineering: bool = False,
                    num_classes: int = 8):
        # embedding_dim：the dimension of word embedding
        # hidden_dim：the dimension of outputs
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.use_gpu = use_gpu

        self.is_reengineering = is_reengineering
        self.word_embeddings = MaskEmbd(vocab_size, embedding_dim , is_reengineering = is_reengineering)
        self.lstm = MaskLstm(embedding_dim, hidden_dim , is_reengineering = is_reengineering)
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = MaskLinear(hidden_dim, label_size , is_reengineering = is_reengineering)
        self.hidden = self.init_hidden()
        if is_reengineering:
            self.module_head = nn.Sequential(
                nn.ReLU(inplace = True),
                nn.Linear(num_classes , 2)
            )

    def init_hidden(self): 
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda()) 
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y  = self.hidden2label(lstm_out[-1])
        if hasattr(self , 'module_head'):
            y = self.module_head(y)
        return y
    
    def get_masks(self):
        masks = {k: v for k, v in self.state_dict().items() if 'mask' in k}
        return masks

    def count_weight_ratio(self):
        # 1. word embedding
        num_wb , size_wb  = self.word_embeddings.count_weight_ratio()
        # 2. LSTM layer
        num_ll , size_ll = self.lstm.count_weight_ratio()
        # 3. Hidden2Label layer
        num_hl , size_hl = self.hidden2label.count_weight_ratio()

        wb_ratio = num_wb / size_wb
        ll_ratio = num_ll / size_ll
        hl_ratio = num_hl / size_hl
        weight_ratio = (num_wb + num_ll + num_hl) / (size_wb + size_ll + size_hl) 

        return weight_ratio , wb_ratio , ll_ratio , hl_ratio

    def get_module_head(self):
        head = {k: v for k, v in self.state_dict().items() if 'module_head' in k}
        return head

def _lstm(embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu , pretrained , is_reengineering = False , is_reengineering_lstm = False) -> LSTMClassifier:
    model = LSTMClassifier(embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu , is_reengineering)
    if pretrained :
        state_dict = model.state_dict() 
        state_dict.update(torch.load('data\\models\\LSTM_classifier.pt'))
        if is_reengineering_lstm:
            state_dict.update(torch.load('data\\binary_classification\\lstm_r8\\tc_0\\lr_head_mask_0.01_0.01_alpha_0.2.pth'))
        model.load_state_dict(state_dict)
    return model

def r8_lstm(*args , **kwargs) -> LSTMClassifier: pass

model_name = 'r8_lstm'
this_module = sys.modules[__name__] 

setattr(this_module , model_name , _lstm) 
