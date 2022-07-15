import mcubes
import numpy as np
import torch
import torch as tc
import open3d as o3d

# from src.shape_reconstruction.utils import network_utils, shape_completion_utils, file_utils
from open3d.cpu.pybind.geometry import PointCloud, TriangleMesh
from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector

from src.shape_reconstruction.utils import network_utils


def train_model(args):
    network_utils.setup_env(args.use_cuda)
    model = network_utils.setup_network(args)
    model.set_mode(train=True)

    net_dump_folder = "networks/"

    file_utils.create_folder("%scheckpoints/" % (net_dump_folder))

    start_epoch = 0
    if args.net_recover:
        start_epoch = network_utils.recover_network(args, model)
    else:
        network_utils.save_network(model, args,
                                   net_dump_folder + "checkpoints/",
                                   "_epoch_" + str(0))

    dataloader = network_utils.load_dataset(args.num_workers,
                                            args.batch_size,
                                            args.mode,
                                            debug=args.debug)
    print("Starting training (%d samples, %d epochs)" %
          (len(dataloader.dataset), args.num_epochs))

    file_utils.create_folder(net_dump_folder + "training_results/")
    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        running_loss = []
        print("EPOCH NUMBER " + str(epoch))
        for idx, data in enumerate(dataloader):
            inputs, targets = network_utils.get_train_data(data, args.use_cuda)
            num_data = inputs.size(0)
            loss = model.train_network(inputs, targets)
            running_loss.append(loss.item() / num_data)
        if (epoch + 1) % args.checkpoint_interval == 0:
            network_utils.save_network(model, args,
                                       net_dump_folder + "checkpoints/",
                                       "_epoch_" + str(int(epoch)))
        file_utils.log_data_to_file(
            np.mean(running_loss), net_dump_folder + "training_results/loss_" +
            args.network_model + ".txt")

    model.eval()

    print("Finished training")

    # Saving final network to file
    print("Saving final network parameters")
    network_utils.save_network(model, args, net_dump_folder + "final_network/")
    tc.cuda.empty_cache()


def test_model(args):
    network_utils.setup_env(args.use_cuda)
    model = network_utils.setup_network(args)
    network_utils.load_parameters(model, args.net_recover_name,
                                  args.network_model, 
                                  hardware = 'gpu' if args.use_cuda else 'cpu')
    model.eval()
    model.apply(network_utils.activate_dropout)
    # When loading a dataset for testing we need to process the data one at a time
    dataloader = network_utils.load_dataset(1, 1, args.mode, debug=args.debug)

    test_cases = shape_completion_utils.get_test_cases(args.test_cases)

    for test_case in test_cases:
        dataloader.dataset.set_model(test_case)
        for idx, data in enumerate(dataloader):
            inputs, targets, observed_pc, object_name, object_pose = network_utils.get_test_data(
                data, args.use_cuda)
            loss, predictions = model.test(inputs,
                                           targets,
                                           num_samples=args.num_test_samples)

            import ipdb; ipdb.set_trace()

            network_utils.save_test_output(predictions, object_name[0],
                                           observed_pc, object_pose, test_case,
                                           args.save_location,
                                           args.network_model,
                                           args.save_voxel_grid,
                                           args.save_samples, args.save_mesh)

            if idx == args.num_objects_to_test:
                break

def voxelize_pc(pc, side=40):
    voxel_size = 1 / side

    ref_idxs = ((pc + 0.5) / voxel_size).long()
    ref_grid = torch.zeros([pc.shape[0], side, side, side], dtype=torch.bool, device=pc.device)

    ref_idxs = ref_idxs.clip(min=0, max=ref_grid.shape[1] - 1)
    ref_grid[
        torch.arange(pc.shape[0]).reshape(-1, 1), ref_idxs[..., 0], ref_idxs[..., 1], ref_idxs[..., 2]] = True

    return ref_grid

def get_bbox_center(pc):
    center = pc.min(0) + (pc.max(0) - pc.min(0)) / 2.0
    return center


def get_diameter(pc):
    diameter = pc.max(0) - pc.min(0)
    return np.max(diameter)



def single_pc_test(args):

    # For now, fix point cloud path JANK

    #observed_pc_path = "/home/fbottarel/workspace/shape_completion_network/datasets/training_data/test_database/partial_pc/pointclouds/002_master_chef_can_pc.pcd"

    # Create a numpy voxelized version of the cloud 

    import os
    # from curvox import pc_vox_utils, binvox_conversions

    patch_size = 40

    observed_pc_path = args.point_cloud_file
    if not os.path.isfile(observed_pc_path):
        print("PC file does not exist")
        return

    observed_pc_np = np.load(observed_pc_path)

    center = get_bbox_center(observed_pc_np)
    diameter = get_diameter(observed_pc_np - center)
    observed_pc_np = (observed_pc_np - center) / diameter

    partial_vox = voxelize_pc(torch.tensor(observed_pc_np).unsqueeze(0), 40).squeeze()

    voxel_x = np.zeros((patch_size, patch_size, patch_size, 1),
                        dtype=np.float32)
    voxel_x[:, :, :, 0] = partial_vox
    input_numpy = np.zeros((1,1,40,40,40), dtype=np.float32)
    input_numpy[0,0,:,:,:] = voxel_x[:, :, :, 0]

    network_utils.setup_env(args.use_cuda)
    model = network_utils.setup_network(args)
    network_utils.load_parameters(model, args.net_recover_name,
                                  args.network_model, 
                                  hardware = 'gpu' if args.use_cuda else 'cpu')
    model.eval()
    model.apply(network_utils.activate_dropout)
    input_tensor = tc.from_numpy(input_numpy)
    if args.use_cuda and tc.cuda.is_available():
        input_tensor = input_tensor.cuda()
    loss, predictions = model.test(input_tensor, 
                         num_samples=args.num_test_samples)

    # Transform voxel into mesh
    predictions = predictions.mean(axis=0)
    predictions = (torch.tensor(predictions) > 0.5).nonzero() / 40
    predictions = predictions - 0.5

    v, t = mcubes.marching_cubes(predictions, 0.5)
    mesh = TriangleMesh(vertices=Vector3dVector(v / 40 - 0.5), triangles=Vector3iVector(t))

    # Get point cluod from reconstucted mesh
    complete = mesh.sample_points_uniformly(number_of_points=8192*2)

    # Visualize
    pc = PointCloud(points=Vector3dVector(reconstruction.cpu().numpy()))
    o3d.visualization.draw([mesh,
                            sampled_pc,
                            pc.paint_uniform_color([1, 0, 0]),
                            PointCloud(points=Vector3dVector(observed_pc_np)).paint_uniform_color([0, 1, 0])])
    # o3d.visualization.draw([mesh])



 