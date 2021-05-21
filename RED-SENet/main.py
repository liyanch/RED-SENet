import os
import argparse
from torch.backends import cudnn
from loader import get_loader
from solver import Solver


def main(args):
    cudnn.benchmark = True

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    if args.result_fig:
        fig_path = os.path.join(args.save_path, 'fig')
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print('Create path : {}'.format(fig_path))

    data_loader = get_loader(mode=args.mode,
                             load_mode=args.load_mode,
                             saved_path=args.saved_path,
                             test_patient=args.test_patient,
                             patch_n=(args.patch_n if args.mode=='train' else None),
                             patch_size=(args.patch_size if args.mode=='train' else None),
                             transform=args.transform,
                             batch_size=(args.batch_size if args.mode=='train' else 1),
                             num_workers=args.num_workers)

    solver = Solver(args, data_loader)
    if args.mode == 'train':
        solver.train(args)
    elif args.mode == 'test':
        solver.test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--load_mode', type=int, default=0)
    parser.add_argument('--data_path', type=str, default='/zengxh_fix/liyanc/heart/25/REDCNN_SENet/img/L001')
    # parser.add_argument('--saved_path', type=str, default='./npy_img/train')
    parser.add_argument('--saved_path', type=str, default='/zengxh_fix/liyanc/heart/25/REDCNN_SENet/npy_img/test')
    parser.add_argument('--save_path', type=str, default='/zengxh_fix/liyanc/heart/25/REDCNN_SENet/save_200epoch/')
    parser.add_argument('--test_patient', type=str, default='L001-test')
    parser.add_argument('--result_fig', type=bool, default=True)

    parser.add_argument('--norm_range_min', type=float, default=0.0)
    parser.add_argument('--norm_range_max', type=float, default=255.0)
    parser.add_argument('--trunc_min', type=float, default=0.0)
    parser.add_argument('--trunc_max', type=float, default=255.0)

    parser.add_argument('--transform', type=bool, default=False)
    # if patch training, batch size is (--patch_n * --batch_size)
    parser.add_argument('--patch_n', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--print_iters', type=int, default=20)
    parser.add_argument('--decay_iters', type=int, default=3000)
    parser.add_argument('--save_iters', type=int, default=20)
    parser.add_argument('--test_iters', type=int, default=8740)

    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--device', type=str)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--multi_gpu', type=bool, default=False)

    args = parser.parse_args()
    print(args)
    main(args)
