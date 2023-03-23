import argparse
import numpy as np
from tqdm import tqdm
from tools.utils import generate_examples, write


parser = argparse.ArgumentParser(description='Data Preparation')

parser.add_argument('-n', '--number_of_training_image',default=200000, type=int,
                    help='Number of training images to be generated')
parser.add_argument('-nl', '--noise_level', default=0.5, type=float
                    help='Level of noise')
parser.add_argument('-e','--envhome', default='', type=str,
                    help='Home directory')
parser.add_argument('-o', '--dataset_out_name', default='train_set', type=str,
                    help='path to output features file')      


def train_set():
    args = parser.parse_args()
    print(args)

    number_of_images = args.number_of_training_image
    noise_level = args.noise_level
    with open(args.envhome + args.dataset_out_name + ".csv", 'w', newline='') as outFile:
        header = ['NAME', 'ROW', 'COL', 'RAD']
        write(outFile, header)
        for i in tqdm(range(number_of_images)):
            img, params = generate_examples(noise_level=noise_level)
            np.save(args.envhome + "datasets/" + args.dataset_out_name + "/" + str(i) + ".npy", img)
            write(outFile, [args.envhome + "datasets/" + args.dataset_out_name + "/" + str(i) + ".npy", params.row, params.col, params.radius])


if __name__ == '__main__':
    train_set()
