import datetime
import os
import pathlib
import re
import time

import torch
from torch.utils.data import DataLoader

from datasets import attack_train_set, test_set
from models import BadNet
from train import eval_badnet, train
from utils import get_parser, select_optimizer

args = get_parser().parse_args()


def main():
    print("{}".format(args).removeprefix('Namespace(').removesuffix(')').replace(', ', '\n'))

    if re.match('cuda:\d', args.device):
        cuda_num = args.device.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_num
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    pathlib.Path("./checkpoints/").mkdir(parents=True, exist_ok=True)
    pathlib.Path("./logs/").mkdir(parents=True, exist_ok=True)

    print(f"\n# load dataset: {args.dataset} ")
    dataset_train, args.output_classes = attack_train_set(is_train=True, args=args)
    dataset_clean, dataset_poisoned = test_set(is_train=False, args=args)

    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers)
    data_loader_clean = DataLoader(dataset_clean, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers)
    data_loader_poisoned = DataLoader(dataset_poisoned, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers)

    model = BadNet(input_channels=dataset_train.channels, output_classes=args.output_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = select_optimizer(args.optimizer, model.parameters(), lr=args.lr)

    basic_model_path = f"./checkpoints/badnet-{args.dataset}.pth"
    start_time = time.time()
    if args.load_local:
        print(f"## Load model from : {basic_model_path}")
        model.load_state_dict(torch.load(basic_model_path), strict=True)
        test_stats = eval_badnet(data_loader_clean, data_loader_poisoned, model, device)
        print(f"Test Clean Accuracy(TCA): {test_stats['clean_acc']:.4f}")
        print(f"Attack Success Rate(ASR): {test_stats['asr']:.4f}")
    else:
        print(f"Start training for {args.epochs} epochs")
        train(data_loader_train, data_loader_clean, data_loader_poisoned, model, criterion,
              optimizer, args.epochs, device, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    main()