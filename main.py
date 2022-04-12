import argparse
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:(lineno)d:%(message)s")

    parser = argparse.ArgumentParser(description='FaceSwap')  # 命令行的注册

    # todo need update
    parser.add_argument('--src_img', required=True, help='Path for source image')

    args = parser.parse_args()


