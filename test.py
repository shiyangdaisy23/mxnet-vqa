import mxnet as mx
import numpy as np
import codecs, json
import os, h5py, sys, argparse
import lstm_feature
import time
import argparse

from dataiter import BucketVQAtestIter




def right_align(seq,lengths):
    v = np.zeros(np.shape(seq))
    N = np.shape(seq)[1]
    for i in range(np.shape(seq)[0]):
        v[i][N-lengths[i]:N-1]=seq[i][0:lengths[i]-1]
    return v


def get_data_test():
    dataset = {}
    test_data = {}
    # load json file
    print('loading json file...')
    with open(input_json) as data_file:
        data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]

    # load image feature
    print('loading image feature...')
    with h5py.File(input_img_h5,'r') as hf:
        tem = hf.get('images_test')
        img_feature = np.array(tem)
    # load h5 file
    print('loading h5 file...')
    with h5py.File(input_ques_h5,'r') as hf:
        # total number of training data is 215375
        # question is (26, )
        tem = hf.get('ques_test')
        test_data['question'] = np.array(tem)-1
        # max length is 23
        tem = hf.get('ques_length_test')
        test_data['length_q'] = np.array(tem)
        # total 82460 img
        tem = hf.get('img_pos_test')
        # convert into 0~82459
        test_data['img_list'] = np.array(tem)-1
        # quiestion id
        tem = hf.get('question_id_test')
        test_data['ques_id'] = np.array(tem)
	# MC_answer_test
	tem = hf.get('MC_ans_test')
	test_data['MC_ans_test'] = np.array(tem)


    print('question aligning')
    test_data['question'] = right_align(test_data['question'], test_data['length_q'])

    print('Normalizing image feature')
    if img_norm:
        tem = np.sqrt(np.sum(np.multiply(img_feature, img_feature), axis=1))
        img_feature = np.divide(img_feature, np.transpose(np.tile(tem,(4096,1))))

    return dataset, img_feature, test_data



parser = argparse.ArgumentParser(description="VQA",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=24,
                    help='the batch size.')

def eval_metrics(): 
    eval_metrics = mx.metric.CompositeEvalMetric()
    #for child_metric in [SmoothL1LossMetric(), AccuracyMetric(), LogLossMetric()]:
    for child_metric in [mx.metric.Accuracy(),mx.metric.CrossEntropy()]:
        eval_metrics.add(child_metric)
    return eval_metrics


####### GLOBAL PARAMETERS ##############
## you can download from https://github.com/VT-vision-lab/VQA_LSTM_CNN Evaluation section ##
input_img_h5 = '/home/ec2-user/workplace/VQA_LSTM_CNN/data_img.h5'
input_ques_h5 = '/home/ec2-user/workplace/VQA_LSTM_CNN/data_prepro.h5'
input_json = '/home/ec2-user/workplace/VQA_LSTM_CNN/data_prepro.json'
img_norm = 1				# normalize the image feature. 1 = normalize, 0 = not normalize


def test(args):
    print 'loading dataset...'
    dataset, img_feature, test_data = get_data_test()
    num_test = test_data['question'].shape[0]
    print(num_test)
    print(list(test_data))
    print(test_data['MC_ans_test'].shape)
    print(test_data['question'][0,:])
    vocabulary_size = len(dataset['ix_to_word'].keys())
    print 'vocabulary_size : ' + str(vocabulary_size)
    
    layout = 'TN'
    buckets = [26]
    current_img_list = test_data['img_list']
    current_imgs = img_feature[current_img_list,:] 
    print(current_imgs.shape)
    print(test_data['question'].shape)
    data_test = BucketVQAtestIter(current_imgs,test_data['question'], args.batch_size, pad = 12, buckets=buckets,
                                            layout=layout)
    sym, arg_params, aux_params = mx.model.load_checkpoint('vqa',200)
    mod = mx.mod.Module(symbol=sym, context=mx.gpu(0),data_names=['text','image'],
                    label_names = ['softmax_label'])
    data_shapes = [mx.io.DataDesc(
                    'text',
                    (seq_len,args.batch_size),
                    layout='TN'),
                   mx.io.DataDesc(
                    'image',
                    (4096,args.batch_size),
                    layout='TN'),
                  
                  ]
    
    label_shapes = [mx.io.DataDesc(
                    'softmax_label',
                    (args.batch_size,),
                    layout='N')]
    print('start')
    mod.bind(data_shapes=data_shapes, label_shapes = label_shapes)
    mod.init_params(arg_params=arg_params, aux_params=aux_params)
    y = mod.predict(data_test)
    y = np.argmax(y.asnumpy(), axis = 1)
    # initialize json list
    result = []
    print(len(y))
    for i in range (0,len(y)):
        ans = dataset['ix_to_ans'][str(y[i]+1)]
        result.append({u'answer': ans, u'question_id': str(test_data['ques_id'][i])})

    # Save to JSON
    print 'Saving result...'
    my_list = list(result)
    dd = json.dump(my_list,open('test_result.json','w'))

if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    
    test(args)

    
