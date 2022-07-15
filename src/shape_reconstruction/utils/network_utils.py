import imp
import os
from collections import OrderedDict

import torch as tc
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

# import dataloader as data_loader
from src.shape_reconstruction.shape_completion import (mc_dropout_network,)


def setup_network(args):

    encoder_decoder = mc_dropout_network.EncoderDecoder(
        args.latent_dim,
        drop_rate=args.dropout_rate,
        use_batch_norm=args.use_batch_norm)
    encoder_decoder.apply(init_weights)
    model = mc_dropout_network.MCDropoutEncoderDecoder(encoder_decoder)

    if args.use_cuda:
        model.set_device("cuda")
        model = model.cuda()
        if tc.cuda.device_count() > 1 and args.train_in_parallel:
            model = nn.DataParallel(model)
    else:
        model.set_device("cpu")

    return model

#
# def load_dataset(batch_size, num_cuda_workers, mode, debug=False, seed=100):
#     # dataloader.dataset.set_model("train_models_train_views")
#     datafile = ""
#     if debug:
#         datafile = "datasets/debug_Dataset.yaml"
#     elif mode == "train":
#         datafile = "datasets/full_Dataset.yaml"
#     elif mode == "test":
#         datafile = "datasets/full_test_Dataset.yaml"
#     else:
#         raise ValueError("No known dataset for " + mode + " mode")
#
#     dataset = ""
#     if mode == "train":
#         shuffle = True
#         dataset = data_loader.trainDataSet(datafile)
#         sampler = None
#     elif mode == "test":
#         shuffle = False
#         dataset = data_loader.testDataSet(datafile)
#         sampler = data_loader.deterRandomSampler(dataset, seed)
#
#     dataloader = DataLoader(dataset,
#                             batch_size=batch_size,
#                             shuffle=shuffle,
#                             num_workers=num_cuda_workers,
#                             pin_memory=True,
#                             sampler=sampler)
#     return dataloader





def get_test_data(data, use_cuda):
    input = data[0].unsqueeze_(1).squeeze(-1)
    target = data[1].unsqueeze_(1).squeeze(-1)
    observed_pc = data[2].squeeze()
    observed_pc = observed_pc.cpu().numpy()
    object_name = data[3]
    object_pose = data[4].squeeze().cpu().numpy()

    targets = []
    if use_cuda:
        inputs = Variable(input).cuda()
        if type(target) != list:
            targets = Variable(target).cuda()
    else:
        inputs = Variable(input)
        if type(target) != list:
            targets = Variable(target)

    return inputs, targets, observed_pc, object_name, object_pose


def get_train_data(data, use_cuda):
    input = data['pc'].unsqueeze_(1).squeeze(-1)
    target = data['truth'].unsqueeze_(1).squeeze(-1)

    targets = []
    if use_cuda:
        inputs = Variable(input).cuda()
        if type(target) != list:
            targets = Variable(target).cuda()
    else:
        inputs = Variable(input)
        if type(target) != list:
            targets = Variable(target)
    return inputs, targets


def setup_env(use_cuda):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"



def load_parameters(network, weight_file, network_model, hardware="cpu"):

    if network_model.lower() == "varley":
        pass
    else:
        #network.load_state_dict(tc.load(weight_file, map_location=hardware))
        network.load_state_dict(tc.load(weight_file, map_location=lambda storage, loc: storage))

    print("Successfully loaded network parameters from file " + weight_file)


def recover_network(args, net):
    print("Recovering network training from checkpoint: %s (%d epochs)" %
          (args.net_recover_name, args.net_recover_epoch))
    load_parameters(net, args.net_recover_name, args.network_model)
    print("Network recovered correctly")
    start_epoch = args.net_recover_epoch
    return start_epoch


def load_dataparallel_on_cpu(savedModel):
    state_dict = tc.load(savedModel)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
        return new_state_dict


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_in')
    elif classname.find('ConvTranspose3d') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_in')


# def to_variable(tensor, use_cuda=True, use_volatile=False, use_asynch=False):
#     if use_cuda and tc.cuda.is_available():
#         if use_volatile:
#             with tc.no_grad():
#                 return Variable(tensor.cuda(async=use_asynch))
#         else:
#             return Variable(tensor.cuda(async=use_asynch))
#     else:
#         return Variable(tensor)


def to_cuda(net, use_cuda=True):
    if use_cuda:
        if tc.cuda.device_count() > 1:
            print("Let's use", tc.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)
            net.cuda()
        if tc.cuda.is_available():
            print("loading network on CUDA")
            net.cuda()
        else:
            print("CUDA not available")
    return net


def activate_dropout(m):
    if type(m) == nn.Dropout or type(m) == nn.Dropout2d or type(
            m) == nn.Dropout3d:
        m.train()
