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
    # center = get_bbox_center(partial)
    # diameter = get_diameter(partial - center)
    # partial = (partial - center) / diameter

    # partial_vox = partial
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
                                   num_samples=10)
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


# def cnn_and_pc_to_mesh(observed_pc,
#                        cnn_voxel,
#                        filepath,
#                        mesh_name,
#                        model_pose,
#                        log_pc=False,
#                        pc_name=""):
#     cnn_voxel = round_voxel_grid_to_0_and_1(cnn_voxel)
#     temp_pcd_handle, temp_pcd_filepath = tempfile.mkstemp(suffix=".pcd")
#     os.close(temp_pcd_handle)
#     pcd = np_to_pcl(observed_pc)
#     pcl.save(pcd, temp_pcd_filepath)
#
#     partial_vox = voxelize_pc(observed_pc[:, 0:3], 40)
#     completed_vox = binvox_rw.Voxels(cnn_voxel, partial_vox.dims,
#                                      partial_vox.translate, partial_vox.scale,
#                                      partial_vox.axis_order)
#     # Now we save the binvox file so that it can be passed to the
#     # post processing along with the partial.pcd
#     temp_handle, temp_binvox_filepath = tempfile.mkstemp(
#         suffix="output.binvox")
#     os.close(temp_handle)
#     binvox_rw.write(completed_vox, open(temp_binvox_filepath, 'w'))
#
#     # This is the file that the post-processed mesh will be saved it.
#     mesh_file = file_utils.create_file(filepath, mesh_name)
#     # mesh_reconstruction tmp/completion.binvox tmp/partial.pcd tmp/post_processed.ply
#     # This command will look something like
#
#     cmd_str = "mesh_reconstruction" + " " + temp_binvox_filepath + " " + temp_pcd_filepath \
#         + " " + mesh_file + " --cuda"
#
#     subprocess.call(cmd_str.split(" "), stdout=FNULL, stderr=subprocess.STDOUT)
#
#     # subprocess.call(cmd_str.split(" "))
#     if log_pc:
#         pcd_file = file_utils.create_file(filepath, pc_name)
#         cmd_str = "pcl_pcd2ply -format 0" + " " + temp_pcd_filepath + " " + pcd_file
#         subprocess.call(cmd_str.split(" "),
#                         stdout=FNULL,
#                         stderr=subprocess.STDOUT)
#         map_object_to_gt(pcd_file, model_pose)
#
#     map_object_to_gt(mesh_file, model_pose)


if __name__ == "__main__":
    patch_size = 40

    parser = argparser.train_test_parser()
    args = parser.parse_args()
    args.network_model = 'checkpoint/mc_dropout_rate_0.2_lat_2000.model'

    model = network_utils.setup_network(args)
    network_utils.load_parameters(model, args.net_recover_name,
                                  args.network_model,
                                  hardware='gpu' if args.use_cuda else 'cpu')

    root = '../../../pcr/data/MCD'
    np.load()

    jaccard = MeanMetric()


    for i, data in tqdm.tqdm(enumerate(ds)):
        if i == 50:
            break

        partial, ground_truth, partial2, ground_truth2 = data

        gt_grid2 = torch.zeros([40, 40, 40])
        gt_grid2[ground_truth2[:, 0], ground_truth2[:, 1], ground_truth2[:, 2]] = 1

        partial_grid2 = torch.zeros([40, 40, 40])
        partial_grid2[partial2[:, 0], partial2[:, 1], partial2[:, 2]] = 1

        out = complete_pc(model, partial)
        grid1 = voxelize_pc2(torch.tensor(out['complete']).unsqueeze(0), 0.025)
        grid2 = voxelize_pc2(torch.tensor(ground_truth).unsqueeze(0), 0.025)
        jaccard(torch.sum(grid1 * grid2, dim=[1, 2, 3]) / torch.sum((grid1 + grid2) != 0, dim=[1, 2, 3]))
        #
        # o3d.visualization.draw([
        #     PointCloud(points=Vector3dVector(out["partial"])).paint_uniform_color([0, 1, 0]),
        #     PointCloud(points=Vector3dVector(out["complete"])).paint_uniform_color([0, 0, 1]),
        #     PointCloud(points=Vector3dVector(ground_truth)).paint_uniform_color([1, 0, 0])],
        #     )
        # o3d.visualization.draw([PointCloud(points=Vector3dVector(partial2)).paint_uniform_color([0, 1, 0]),
        #     PointCloud(points=Vector3dVector(ground_truth2)).paint_uniform_color([0, 0, 1]), TriangleMesh.create_coordinate_frame(origin=[40, 00, 40], size=40.),
        #                         TriangleMesh.create_coordinate_frame(origin=[0, 0, 0], size=40.)
        # ])

    print(jaccard.compute())

