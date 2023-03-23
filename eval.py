import argparse
import numpy as np
import torch
from model.circle_detector import Net
import torchvision.transforms as transforms
from tools.utils import CircleParams, iou, generate_examples
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Model Evaluation')

parser.add_argument('-n', '--number_of_images',default=1000, type=int,
                    help='Number of test images to be generated')
parser.add_argument('-nl', '--noise_level', default=0.5, type=float
                    help='Level of noise')
parser.add_argument('-m','--model_path', default='model_eval.pth', type=str,
                    help='Path to the saved model')
     

def main():
    args = parser.parse_args()
    print(args)
    accAt70 = [] 
    accAt80 = []  
    accAt90 = []
    accAt95 = [] 

    for _ in tqdm(range(args.number_of_images)):
        img, params = generate_examples(img_size=200,noise_level=args.noise_level,max_radius=50)
        detected = find_circle(img,args)
        val = iou(params, detected)
        # print(val)
        accAt70.append((val > 0.7))
        accAt80.append((val > 0.8))
        accAt90.append((val > 0.9))
        accAt95.append((val > 0.95))
        # print()

    print("Accuracy at IoU Threshold 0.7(%f %) \
          \nAccuracy at IoU Threshold 0.8(%f %) \
          \nAccuracy at IoU Threshold 0.9(%f %) \
          \nAccuracy at IoU Threshold 0.95(%f %)",
          (sum(accAt70) / args.number_of_images) * 100, (sum(accAt80) / args.number_of_images) * 100, (sum(accAt90) / args.number_of_images) * 100, (sum(accAt95) / args.number_of_images) * 100 )

def find_circle(img,args):
    model = Net()
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        image = np.expand_dims(np.asarray(img), axis=0)
        image = torch.from_numpy(np.array(image, dtype=np.float32))
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        image = normalize(image)
        image = image.unsqueeze(0)
        output = model(image)

    params =  [round(i) for i in (200 * output).tolist()[0]]
    return CircleParams(params[0],params[1],params[2])

if __name__=='__main__':
    main()