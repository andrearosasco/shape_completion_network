from src.shape_reconstruction.utils import argparser
import shape_completion
if __name__ == "__main__":
    parser = argparser.train_test_parser()
    args = parser.parse_args()

    args.point_cloud_file = 'assets/partial_bleach_317.npy'
    args.network_model = 'checkpoint/mc_dropout_rate_0.2_lat_2000.model'
    shape_completion.single_pc_test(args)

