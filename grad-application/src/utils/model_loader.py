import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name, num_classes=20 , g = None , input_dim = None , output_dim = None , conv = None):
    if model_name == 'GCN':
        from models.gcn import Classifer
        model = Classifer(g,input_dim, num_classes,conv)
    else:
        raise ValueError
    return model


def load_trained_model(model_name, n_classes, trained_model_path):
    model = load_model(model_name, num_classes=n_classes)
    model.load_state_dict(torch.load(trained_model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model
