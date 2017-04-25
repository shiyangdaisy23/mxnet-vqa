# mxnet-vqa

## Requirements

This code is written in Python and requires [MXNET](http://mxnet.io/). The preprocssinng code is in Python.

## Data Preprocessing
Here we list two preprocessed data using VQA v1.0 dataset

#### DATA ONE
Download preprocessed data from [VQA_LSTM_CNN](https://github.com/VT-vision-lab/VQA_LSTM_CNN)

Under Evaluation section, you can download the features. (Train on train set and evaluate on validation set)
You will see three files in the folder: `data_prepro.h5`, `data_prepro.json` and `data_img.h5`.
`data_prepro.h5` contains questions and answers for train and test sets. `data_prepro.json` contains index map for all words in questions and answers. `data_img.h5` contains image features using pretrained VGG19 network. Image feature size is 4096.


#### DATA TWO
Download original text datasets(annotation and question) from [VQA](http://www.visualqa.org/vqa_v1_download.html) and run

```
$ python textpreprocess.py
```
to get `vqa_raw_train.json`, `vqa_raw_test.json` and `vqa_raw_val.json`.

Once you have these, run

```
$ python prepro.py --input_train_json vqa_raw_train.json --input_val_json vqa_raw_val.json --input_test_json vqa_raw_test.json --num_ans 1000
```

to get the question features. `--num_ans` specifiy how many top answers you want to use during training. This will generate two files in your main folder, `data_prepro.h5` and `data_prepro.json`. `data_prepro.h5` contains questions and answers for train, validation and test sets. `data_prepro.json` contains index map for all words in questions and answers. 

We use pretrained resnet-152 network to get the image features. Please refer to [VQA-MCB](https://github.com/akirafukui/vqa-mcb/tree/master/preprocess). After preprocessing, you should have processed image features stored in .jpg.npz files. Image feature size is 2048\*14\*14

## Training
### Basic model
In the basic model, we just concatenate the text and image features. Reference paper is [VQA:Visual Question Answering](https://arxiv.org/abs/1505.00468). (We use DATA ONE) Run
```
$ python basic_train.py
```
You can also change the model and only use text or image features for training.
### Tensor Sketching model
In this model, we add higher order correlation between text and image features to the network. We use tensor sketching to preserve the higher order correlation and also keep reasonable computation complexity. Reference paper is [Compact Bilinear Pooling](https://arxiv.org/abs/1511.06062). (We use DATA ONE) Run
```
$ python ts_train.py
```
### Tensor Sketching with Attention model
TO BE DONE........
## Testing
After training, you should have saved the model parameters in a .params file. Here we just load the model.
Run
```
$ python test.py
```
This will generate a .json file. Run 
```
$ python s2i.py
```
to make the result file readable to [VQA Evaluation Tool](https://github.com/VT-vision-lab/VQA/). Then you can use the [VQA Evaluation Tool](https://github.com/VT-vision-lab/VQA/) to evaluate.
