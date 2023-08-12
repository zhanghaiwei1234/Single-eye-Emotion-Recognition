# Instructions 
Our SEE dataset provides both intensity frames and accumulated event frames. The event accumulation period is 33 ms, corresponding to the event-based camera's ASP frame rate, i.e., 30 FPS. 

## The file structure is shown as follows:
```
SEE/
|-- event
|   |-- angry
|   |   |-- 1_001_man_24_Master_normal_angry_45_take000
|   |   |   |-- 00000.jpg
|   |   |   |-- 00001.jpg
|   |   |   |-- ...
|   |   |   |-- ...
|   |   |   |-- ...
|   |   |-- 2_001_man_24_Master_normal_angry_50_take001
|   |   |-- ...
|   |   |-- ...
|   |   `-- ...
|   |-- disgust
|   |-- fear
|   |-- happiness
|   |-- neutraL
|   |-- sadness
|   `-- surprise
`--  frame
    |-- angry
    |-- disgust
    |-- fear
    |-- happiness
    |-- neutraL
    |-- sadness
    `-- surprise
```

## Sequence Name Format
For each sequence, there are two folders: one is under the frame folder, containing the intensity frames; the other is under the event folder, providing the corresponding accumulated event frames. The two folders have the same name, with the following format:
```
[sequence#]_[ID]_[sex]_[age]_[degree]_[lighting-condition]_[emotion]_[length]_[take#]
```
>[sequence#]: the sequence number
[ID]: volunteer ID
[sex]: volunteer's sex, man or woman
[age]: volunteer's age
[degree]: volunteer's highest educational degree
[lighting-condition]: the lighting condition of the sequence
[emotion]: the ground truth label of the sequence
[length]: the length of the sequence in the number of frames
[take#]: the sequence is the [take#]th sequence of the volunteer with [ID]


For example:
``` 
1_001_man_24_Master_normal_angry_45_take000 
```

>[sequence#=1]: the first sequence
[ID=001]: volunteer ID is 001
[sex=man]: a **male** volunteer
[age=24]: volunteer's age is 24 years old
[degree=Master]: volunteer's highest educational degree is a master degree
[lighting-condition=normal]: the sequence is recorded under **normal lighting condition**
[emotion=angry]: the ground truth label of the sequence is **angry**
[length=45]: total number of frames in the sequence is 45
[take#]: It is the first sequence of the volunteer 001.

## Dataset Training/testing split
The training/testing split is provided in the emotion.json
One example is shown as follows:
```
"1011_035_man_20_undergraduate_normal_angry_63_take000": {
			"subset": "training",
			"annotations": {
				"label": "angry",
				"segment": [0, 63]
			}
		},
```
>[subset: training]: indicates the seqeuce belongs to __training set__
>[label: angry]: indicates ground truth label of the sequence is __angry__
>[segment: [0,63] ]: indicates the **start** and **end** frame index of the sequence

