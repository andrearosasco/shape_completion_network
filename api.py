import os
import time

import mcubes
import numpy as np
import torch
import tqdm
from open3d.cpu.pybind.geometry import TriangleMesh, PointCloud, VoxelGrid
from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector
from torchmetrics import MeanMetric

from datasets.GraspingDataset import GraspingDataset
from src.shape_reconstruction.shape_completion.shape_completion import get_bbox_center, get_diameter, voxelize_pc
from src.shape_reconstruction.utils import network_utils, argparser
import open3d as o3d


def complete_pc(model, partial):
    partial_vox = voxelize_pc(torch.tensor(partial).unsqueeze(0), 40).squeeze()
    voxel_x = np.zeros((patch_size, patch_size, patch_size, 1),
                       dtype=np.float32)

    voxel_x[:, :, :, 0] = partial_vox
    input_numpy = np.zeros((1, 1, 40, 40, 40), dtype=np.float32)
    input_numpy[0, 0, :, :, :] = voxel_x[:, :, :, 0]

    model.eval()
    model.apply(network_utils.activate_dropout)
    input_tensor = torch.from_numpy(input_numpy)
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
    # start = time.perf_counter()
    loss, predictions = model.test(input_tensor,
                                   num_samples=20)
    # model.sample_
    # print(time.perf_counter() - start)

    # Transform voxel into mesh
    predictions = predictions.mean(axis=0)
    v, t = mcubes.marching_cubes(predictions, 0.5)
    mesh = TriangleMesh(vertices=Vector3dVector(v / 40 - 0.5), triangles=Vector3iVector(t))
    #
    # # Get point cloud from reconstructed mesh
    complete = mesh.sample_points_uniformly(number_of_points=8192 * 2)
    complete = np.array(complete.points)

    ret = {"partial": partial,
            "complete": complete,
            "mesh": mesh}

    return ret


def voxelize_pc2(pc, voxel_size):
    side = int(1 / voxel_size)

    ref_idxs = ((pc + 0.5) / voxel_size).long()
    ref_grid = torch.zeros([pc.shape[0], side, side, side], dtype=torch.bool, device=pc.device)

    ref_idxs = ref_idxs.clip(min=0, max=ref_grid.shape[1] - 1)
    ref_grid[
        torch.arange(pc.shape[0]).reshape(-1, 1), ref_idxs[..., 0], ref_idxs[..., 1], ref_idxs[..., 2]] = True

    return ref_grid


if __name__ == "__main__":
    patch_size = 40

    parser = argparser.train_test_parser()
    args = parser.parse_args()
    args.network_model = 'checkpoint/mc_dropout_rate_0.2_lat_2000.model'

    model = network_utils.setup_network(args)
    network_utils.load_parameters(model, args.net_recover_name,
                                  args.network_model,
                                  hardware='gpu' if args.use_cuda else 'cpu')

    partial = np.load('assets/partial_bleach_317.npy')

    center = get_bbox_center(partial)
    diameter = get_diameter(partial - center)
    normalized_partial = (partial - center) / diameter

    out = complete_pc(model, normalized_partial)

    reconstruction = out['complete'] * diameter + center

    o3d.visualization.draw([
        PointCloud(points=Vector3dVector(partial)).paint_uniform_color([0, 1, 0]),
        PointCloud(points=Vector3dVector(reconstruction)).paint_uniform_color([0, 0, 1])],
        )

