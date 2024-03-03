import argparse
import os.path
import sys
import torch
from reengineer import Reengineer
from torch.utils.data import DataLoader
import copy
import utils.DataProcessing as DP
sys.path.append('../')
sys.path.append('../..')
sys.path.append('seam-application\\src')
import reengineer
from config import load_config
from tqdm import tqdm
from models import lstm
from models.lstm import r8_lstm



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['lstm'] , default='lstm')
    parser.add_argument('--dataset', type=str, choices=['r8'] , default='r8')
    parser.add_argument('--target_class', type=int, default=0)
    parser.add_argument('--shots', default=-1, type=int, help='how many samples for each classes.')
    parser.add_argument('--seed', type=int, default=0, help='the random seed for sampling ``num_classes'' classes as target classes.')
    parser.add_argument('--n_epochs', type=int, default=300)

    parser.add_argument('--lr_head', type=float, default=0.1)
    parser.add_argument('--lr_mask', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='the weight for the weighted sum of two losses in re-engineering.')
    parser.add_argument('--early_stop', type=int, default=-1)
    parser.add_argument('--tuning_param', action='store_true')

    args = parser.parse_args()
    return args


#lr_mask：0.01，alpha：1
def reengineer(model, train_loader, test_loader, lr_mask, lr_head, n_epochs, alpha, early_stop, acc_pre_model , target_class):
    #data/binary_class/vgg_cifar_10/tc_0
    save_dir = f'{config.project_data_save_dir}/{args.model}_{args.dataset}/tc_{args.target_class}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = f'{save_dir}/lr_head_mask_{args.lr_head}_{args.lr_mask}_alpha_{args.alpha}.pth'

    reengineer = Reengineer(model, train_loader, test_loader, acc_pre_model=acc_pre_model)
    reengineered_model = reengineer.alter(lr_mask=lr_mask, lr_head=lr_head,
                              n_epochs=n_epochs, alpha=alpha, early_stop=early_stop , target_class=target_class)


    masks = reengineered_model.get_masks()
    module_head = reengineered_model.get_module_head()
    masks.update(module_head)
    # save the best module on target class
    torch.save(masks, save_path)

    # check
    model_static = model.state_dict()
    reengineered_model_static = reengineered_model.state_dict()
    for k in model_static:
        if 'mask' not in k and 'module_head' not in k:
            model_weight = model_static[k]
            reengineered_model_weight = reengineered_model_static[k]
            assert (model_weight == reengineered_model_weight).all()


@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()
    n_corrects = 0
    n_samples = 0

    for batch_inputs, batch_labels in tqdm(test_loader, ncols=80, desc=f'Eval '):
        batch_inputs, batch_labels = batch_inputs.to('cuda'), batch_labels.to('cuda')
        batch_labels = torch.squeeze(batch_labels)
        # match the size
        model.batch_size = len(batch_labels) 
        model.hidden = model.init_hidden() # 防止最后的batch_size不符合

        batch_outputs = model(batch_inputs.t()) # t():转置
        n_samples += batch_labels.shape[0]
        batch_preds = torch.argmax(batch_outputs, dim=1)
        if args.target_class >= 0:
            batch_preds = batch_preds == args.target_class
            batch_labels = batch_labels == args.target_class
        n_corrects += torch.sum(batch_preds == batch_labels).item()

    acc = float(n_corrects) / n_samples
    return acc


def eval_pretrained_model():
    dtest_set = DP.TxtDatasetProcessing(DATA_DIR, TEST_DIR, TEST_FILE, TEST_LABEL, sentence_len, corpus , args.target_class , reorganize = False)
    test_loader = DataLoader(dtest_set,
                          batch_size = batch_size,
                          shuffle=False,
                          num_workers=4
                         )
    acc = evaluate(model, test_loader)
    return acc


def main():

    acc_pre_model = eval_pretrained_model()
    print(f'\nPretrained Model Test Acc: {acc_pre_model:.2%}\n\n')

    model_ = eval(f'{args.dataset}_{args.model}')(embedding_dim = embedding_dim , use_gpu = use_gpu ,
                                                  hidden_dim = hidden_dim ,vocab_size = len(corpus.dictionary) ,
                                                  label_size = 8 , batch_size = batch_size ,
                                                  pretrained=True, is_reengineering=True , is_reengineering_lstm = True).to('cuda')
    dtrain_set = DP.TxtDatasetProcessing(DATA_DIR, TRAIN_DIR, TRAIN_FILE, TRAIN_LABEL, sentence_len, corpus, args.target_class , reorganize = True)

    train_loader = DataLoader(dtrain_set,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4
                         )
    dtest_set = DP.TxtDatasetProcessing(DATA_DIR, TEST_DIR, TEST_FILE, TEST_LABEL, sentence_len, corpus , args.target_class , reorganize = True)
    test_loader = DataLoader(dtest_set,
                          batch_size = batch_size,
                          shuffle=False,
                          num_workers=4
                         )

    reengineer(model_, train_loader, test_loader, args.lr_mask, args.lr_head,
                  args.n_epochs, args.alpha, args.early_stop, acc_pre_model , args.target_class)

    print(f'\nPretrained Model Test Acc: {acc_pre_model:.2%}\n\n')

DATA_DIR = 'D:\\Lab\\BUAA\\Demo\\data\\dataset\\r8'
TRAIN_DIR = 'D:\\Lab\\BUAA\\Demo\\data\\dataset\\r8\\train_txt'
TEST_DIR = 'D:\\Lab\\BUAA\\Demo\\data\\dataset\\r8\\test_txt'
TRAIN_FILE = 'train_txt.txt'
TEST_FILE = 'test_txt.txt'
TRAIN_LABEL = 'train_label.txt'
TEST_LABEL = 'test_label.txt'


if __name__ == '__main__':
    args = get_args()
    print(args)
    config = load_config()

    num_workers = 4
    pin_memory = True

    model_name = args.model

    embedding_dim = 100
    hidden_dim = 50
    sentence_len = 32
    train_file = os.path.join(DATA_DIR, TRAIN_FILE)
    test_file = os.path.join(DATA_DIR, TEST_FILE)
    fp_train = open(train_file, 'r')
    train_filenames = [os.path.join(TRAIN_DIR, line.strip()) for line in fp_train]
    filenames = copy.deepcopy(train_filenames)
    fp_train.close()
    fp_test = open(test_file, 'r')
    test_filenames = [os.path.join(TEST_DIR, line.strip()) for line in fp_test]
    fp_test.close()
    filenames.extend(test_filenames)

    corpus = DP.Corpus(DATA_DIR, filenames)
    nlabel = 8
    batch_size = 5
    use_gpu = torch.cuda.is_available()
    

    ### create model
    model =  eval(f'{args.dataset}_{args.model}')(embedding_dim = embedding_dim , 
                                                  hidden_dim = hidden_dim ,
                                                  vocab_size = len(corpus.dictionary) , label_size = 8,
                                                  batch_size = batch_size , use_gpu = use_gpu , pretrained=True).to('cuda')
    main()