# Deep-Learning-Final-Project

## Abstract
In the field of human-computer interaction, it is important to predict how to predict a person’s object manipulation actions in 3D space in an egocentric way. There has been a lot of related work in recent years to make progress in this area. We use a baseline approach using recurrent neural networks. However, the conventional approach using the same model to train the two steps of grasping and fetching cannot achieve good results, while we use two different models to train the two steps sepa- rately and achieve better results. Training on a large multimodal dataset consisting of more than 1 million frames of RGB-D and IMU streams.

## Environment setup
This code has been tested on Ubuntu 20.04, Python 3.7.0, Pytorch 1.9.0, CUDA 11.2.
Please install related libraries before running this code. The detailed information is included in `./requirement.txt`.

## Train

### Prepare training datasets

Download the datasets from [here](https://drive.google.com/drive/folders/1Eszuzqg0mnGX8fwUjC0YUZbPqtP5xJYp?usp=sharing) and put it into `./benchmark/`.

#### Dataset folder hierarchy
```bash
Dataset/
    ├──annotrain/ # The annotation of train set
        ├── bathroomCabinet/ # The name of different scenes
            ├── bathroomCabinet_1.txt/ # The groundtruth of each clip
            ├── bathroomCabinet_2.txt/
            ├── bathroomCabinet_3.txt/
            └── bathroomCabinet_4.txt/
                
        ├── bathroomCounter/ 
        ├── ...
        └── nightstand/
    ├──annonoveltest_final/ # The annotation of test set (unseen scenes)
        └── ...
    ├──annotest_final/ # The annotation of test set (seen scenes)
    ├──annovalidate_final/ # The annotation of validation set
    ├──sequences/ # The pointcloud and imu data of each scene
        ├── bathroomCabinet/ 
            ├── bathroomCabinet_1/ 
                 ├── pointcloud/ # The pointcloud files
                 ├── transformation/ # The odometry files
                 └── data.txt/ # The imu data of bathroomCabinet_1
            .
            .
    
            └── bathroomCabinet_6/
        .
        .
    
        └── woodenTable
```

### Train a model
To train the predictor model, run `train.py` with the desired configs:

```
python ./train_two.py --data_path ./Dataset
```

## Test
Copy the grab and release models in experiment folder into the main directory. Name them as "{your name}_grab/release.pth". (LSTM_grab.pth and LSTM_release in the given example)


```
python test.py 	                          \
	--model_name two           \ # tracker_name
	--checkpoint ./experiment/LSTM   #model_path
	--datapath ./data_path   #data_path
```

The testing result will be saved in the `./results/model_name` directory.


## Evaluation


```
python eval.py 	                          \
	--data_path  ./Dataset         \ # The path of the dataset
	--model_name two   # The name of predictor
```

The results will be saved in the `./results/` directory.