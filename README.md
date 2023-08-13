# In the Blink of an Eye: Event-based Emotion Recognition
Haiwei Zhang, [Jiqing Zhang](https://zhangjiqing.com), [Bo Dong](https://dongshuhao.github.io/), Pieter peers, Wenwei Wu, Xiaopeng Wei, Felix Heide, [Xin Yang](https://xinyangdut.github.io/)

[[paper](https://doi.org/10.1145/3588432.3591511)] [[dataset](http://www.dluticcd.com/)]

<img width="100%" src="https://github.com/zhanghaiwei1234/Single-eye-Emotion-Recognition/blob/main/img/introduce.png"></a>
Demonstration of a wearable single-eye emotion recognition prototype system consisting with a bio-inspired event-based camera (DAVIS346) and a low-power NVIDIA Jetson TX2 computing device. Event-based cameras simultaneously provide intensity and corresponding events, which we input to a newly designed lightweight Spiking Eye Emotion Network (SEEN) to effectively extract and combine spatial and temporal cues for emotion recognition. Given a sequence, SEEN takes the start and end intensity frames (green boxes) along with $n$ intermediate event frames (red boxes) as input. Our prototype system consistently recognizes emotions based on single-eye areas under different lighting conditions at $30$ FPS. 

## Abstract
We introduce a wearable single-eye emotion recognition device and a real-time approach to recognizing emotions from partial observations of an emotion that is robust to changes in lighting conditions. At the heart of our method is a bio-inspired event-based camera setup and a newly designed lightweight Spiking Eye Emotion Network (SEEN). Compared to conventional cameras, event-based cameras offer a higher dynamic range (up to 140 dB vs. 80 dB) and a higher temporal resolution (in the order of $\mu$s vs. 10s of $m$s). Thus, the captured events can encode rich temporal cues under challenging lighting conditions. However, these events lack texture information, posing problems in decoding temporal information effectively. SEEN tackles this issue from two different perspectives. First, we adopt convolutional spiking layers to take advantage of the spiking neural network's ability to decode pertinent temporal information. Second, SEEN learns to extract essential spatial cues from corresponding intensity frames and leverages a novel weight-copy scheme to convey spatial attention to the convolutional spiking layers during training and inference. We extensively validate and demonstrate the effectiveness of our approach on a specially collected Single-eye Event-based Emotion (SEE) dataset. To the best of our knowledge, our method is the first eye-based emotion recognition method that leverages event-based cameras and spiking neural networks.

## Our dataset SEE

To address this lack of training data for event-based emotion recognition, we collect a new Single-eye Event-based Emotion (SEE) dataset;  SEE contains data from 111 volunteers captured with a DAVIS346 event-based camera placed in front of the right eye and mounted on a helmet; SEE contains videos of 7 emotions under four different lighting conditions: normal, overexposure, low-light, and high dynamic range (HDR) (Figure 3(a)). The average video length ranges from 18 to 131 frames, with a mean frame number of 53.5 and a standard deviation of 15.2 frames, reflecting the differences in the duration of emotions between subjects. In total, SEE contains 2, 405/128, 712 sequences/frames with corresponding raw events for a total length of 71.5 minutes (Figure 3(b)), which we split in 1, 638 and 767 sequences for training and testing, respectively.
<img width="100%" src="https://github.com/zhanghaiwei1234/Single-eye-Emotion-Recognition/blob/main/img/dataset.png"></a>

Our dataset can be found [here](http://www.dluticcd.com/). Our dataset detail instructions can be find at [here](https://github.com/zhanghaiwei1234/Single-eye-Emotion-Recognition/blob/main/Dataset_Instructions.md)

## Requirements
You can use our Dockerfile to create a docker image with all the necessary tools. Download our [Dockerfile](https://pan.baidu.com/s/1L8QyWTNK1JKdimNwLYI1uQ) (password: seen). 
1. To build the docker image run: ```docker load -i SEEN.tar```

2.Create a container run: ```docker run --gpus all -dit --name your_container_name -v File_path_to_be_mounted -p 7000:80 --shm-size=10G 
load_image_id /bin/bash ```

3. Start container run: ```docker start your_container_name```

4. Entry container run: ```sudo docker exec -it your_container_name bash```

## About coda



## Training

1. Download our SEE dataset 

2. Change dataset path at line 23-25 in opts.py. 

3. run ``` CUDA_VISIBLE_DEVICES=0  python main.py   --result_path  your_save_path   --inference  --tensorboard --sample_duration 4    --sample_t_stride 4  --inference_sample_duration 4  ```

## Testing 

1. Change test checkpoint path at line 16 in opts.py. 

2. run ``` CUDA_VISIBLE_DEVICES=0  python evaluation.py   --result_path  your_save_path   --inference  --tensorboard --sample_duration 4    --sample_t_stride 4  --inference_sample_duration 4  ```
    - The predicted results will be saved in  your_save_path.
3. run ``` python read_20json_results.py ``` to calculate UAR, WAR, accuracy of different emotion classes, and accuracy under different light conditions. You can also observe the loss and precision of the training process through Tensorboard.
