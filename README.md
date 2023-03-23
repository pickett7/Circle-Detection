# Circle-Detection

Deep learning Convolution network for detecting circle in a image with arbitrary noise. The output of the network are position of the circle and its radius. A simple network with convolution layers followed by fully connected layers is implemented in Pytorch.

## Model

<img width="902" alt="image" src="https://user-images.githubusercontent.com/45058906/227034313-13c3efa7-68e7-42ee-a030-ffaf913671af.png">

## Usage

### Data preparation

`data_prep.py -n <number_of_training_image_to_be_generated> -nl <noise_level>`

### Training

`train.py -b <batch_size> `

### Validation

`eval.py -n <number_of_images_for_evaluation>`

## Evaluation

Network is trained with 50000 generated images in GPU provided by Google Collab(Free). Model is trained with images with noise level 0.5. All the parametres are configurable.

### Model performance
Accuracy @ IoU thresholds at 0.7, 0.8, 0.9, 0.95 

<img width="582" alt="image" src="https://user-images.githubusercontent.com/45058906/227056151-d7cca9f3-7d7b-4f79-a37c-97170f644703.png">






