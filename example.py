import argparse
from main import get_args_parser, inference
from pathlib import Path

if __name__ == '__main__':
    # parser = argparse.ArgumentParser('DETR training and evaluation script',
    #                                  parents=[get_args_parser()])
    # args = parser.parse_args()
    # print(args)
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model = inference()
    print(model)
    # print(get_args_parser())
