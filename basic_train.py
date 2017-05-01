import mxnet as mx
import numpy as np
import codecs, json
import os, h5py, sys, argparse
import time
import argparse
import logging

parser = argparse.ArgumentParser(description="VQA",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--test', default=False, action='store_true',
                    help='whether to do testing instead of training')
parser.add_argument('--model-prefix', type=str, default=None,
                    help='path to save/load model')
parser.add_argument('--load-epoch', type=int, default=0,
                    help='load from epoch')
parser.add_argument('--num-layers', type=int, default=2,
                    help='number of stacked RNN layers')
parser.add_argument('--num-hidden', type=int, default=1024,
                    help='hidden layer size')
parser.add_argument('--num-embed', type=int, default=128,
                    help='embedding layer size')
parser.add_argument('--gpus', type=str, default = '0',
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu. ' \
                         'Increase batch size when using multiple gpus for best performance.')
parser.add_argument('--kv-store', type=str, default='device',
                    help='key-value store type')
parser.add_argument('--num-epochs', type=int, default=25,
                    help='max num of epochs')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='the optimizer type')
parser.add_argument('--mom', type=float, default=0.0,
                    help='momentum for sgd')
parser.add_argument('--wd', type=float, default=0.00001,
                    help='weight decay for sgd')
parser.add_argument('--batch-size', type=int, default=30,
                    help='the batch size.')
parser.add_argument('--disp-batches', type=int, default=50,
                    help='show progress for every n batches')
# When training a deep, complex model, it's recommended to stack fused RNN cells (one
# layer per cell) together instead of one with all layers. The reason is that fused RNN
# cells doesn't set gradients to be ready until the computation for the entire layer is
# completed. Breaking a multi-layer fused RNN cell into several one-layer ones allows
# gradients to be processed ealier. This reduces communication overhead, especially with
# multiple GPUs.
parser.add_argument('--stack-rnn', default=False,
                    help='stack fused RNN cells to reduce communication overhead')


def right_align(seq,lengths):
    v = np.zeros(np.shape(seq))
    N = np.shape(seq)[1]
    for i in range(np.shape(seq)[0]):
        v[i][N-lengths[i]:N-1]=seq[i][0:lengths[i]-1]
    return v

def get_data():
    dataset = {}
    train_data = {}
    # load json file
    print('loading json file...')
    with open(input_json) as data_file:
        data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]

    # load image feature
    print('loading image feature...')
    with h5py.File(input_img_h5,'r') as hf:
        # -----0~82459------
        tem = hf.get('images_train')
        img_feature = np.array(tem)
    # load h5 file
    print('loading h5 file...')
    with h5py.File(input_ques_h5,'r') as hf:
        # total number of training data is 215375
        # question is (26, )
        tem = hf.get('ques_train')
        train_data['question'] = np.array(tem)-1
        # max length is 23
        tem = hf.get('ques_length_train')
        train_data['length_q'] = np.array(tem)
        # total 82460 img
        tem = hf.get('img_pos_train')
	  # convert into 0~82459
        train_data['img_list'] = np.array(tem)-1
        # answer is 1~1000
        tem = hf.get('answers')
        train_data['answers'] = np.array(tem)-1

    print('question aligning')
    train_data['question'] = right_align(train_data['question'], train_data['length_q'])

    print('Normalizing image feature')
    if img_norm:
        tem = np.sqrt(np.sum(np.multiply(img_feature, img_feature), axis=1))
        img_feature = np.divide(img_feature, np.transpose(np.tile(tem,(4096,1))))

    return dataset, img_feature, train_data

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

def eval_metrics(): 
    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [mx.metric.Accuracy(),mx.metric.CrossEntropy()]:
        eval_metrics.add(child_metric)
    return eval_metrics

def evaluation_callback(iter_no, sym, arg, aux):
    if iter_no % 20 == 0:
        mx.model.save_checkpoint('vqa_eva', iter_no, sym, arg, aux)
    if iter_no == 399:
        mx.model.save_checkpoint('vqa_eva', iter_no, sym, arg, aux)
####### GLOBAL PARAMETERS ##############
## you can download from https://github.com/VT-vision-lab/VQA_LSTM_CNN Evaluation section ##
input_img_h5 = '/home/ec2-user/workplace/VQA_LSTM_CNN/data_img.h5'
input_ques_h5 = '/home/ec2-user/workplace/VQA_LSTM_CNN/data_prepro.h5'
input_json = '/home/ec2-user/workplace/VQA_LSTM_CNN/data_prepro.json'
img_norm = 1				# normalize the image feature. 1 = normalize, 0 = not normalize


def train(args):
    logging.basicConfig(filename='out.log', level=logging.INFO)
    logging.info('Started')
    
    print 'loading dataset...'
    dataset, img_feature, train_data = get_data()
    num_train = train_data['question'].shape[0]
    vocabulary_size = len(dataset['ix_to_word'].keys())
    print(list(dataset))
    print(type(dataset['ix_to_word']))
    print 'vocabulary_size : ' + str(vocabulary_size)
    print 'number of question :' + str(num_train)
    
    ##################### DATA ITERATOR ###########################
    ##### We use 5000 training samples as evaluation sets
    ############################################################
    layout = 'TN'
    buckets = [26]
    current_img_list = train_data['img_list']
    current_imgs = img_feature[current_img_list,:] 
    print(current_imgs.shape)
    print(train_data['answers'].shape)
    evaluation_num = 5000
    eva_idx = np.random.choice(train_data['answers'].shape[0], evaluation_num, replace=False)
    train_idx = list(set(np.arange(train_data['answers'].shape[0]))-set(eva_idx))
    train_img = current_imgs[train_idx,...]
    train_que = train_data['question'][train_idx,...]
    train_ans = train_data['answers'][train_idx,...]
    eva_img = current_imgs[eva_idx,...]
    eva_que = train_data['question'][eva_idx,...]
    eva_ans = train_data['answers'][eva_idx,...]
    data_train  = mx.rnn.BucketSentenceIter(train_img, train_que, train_ans, args.batch_size, buckets=buckets,layout=layout)
    data_eva  = mx.rnn.BucketSentenceIter(eva_img, eva_que, eva_ans, args.batch_size, buckets=buckets,layout=layout)
    
    ################# MODULE #######################
    ###VQA model with MCB:based on https://arxiv.org/pdf/1606.01847.pdf
    ################################################
    seq_len = 26
 
    data = mx.sym.Variable('text')
    label = mx.sym.Variable('softmax_label')
    embed = mx.sym.Embedding(data=data, input_dim=vocabulary_size, output_dim=args.num_embed,name='embed')
    img_data = mx.sym.Variable('image')
    img_data = mx.sym.transpose(img_data)
    
    cell = mx.rnn.FusedRNNCell(args.num_hidden, num_layers=args.num_layers, mode='lstm')
    cell.reset()
    output, _ = cell.unroll(seq_len, inputs=embed, merge_outputs=True, layout='TNC')
    output = mx.sym.SequenceLast(data = output)                
    pred = mx.sym.Reshape(output, shape=(-1, args.num_hidden),name='text_lstm')
    
    
    pred = mx.sym.Concat(pred,img_data,dim = 1)
    pred = mx.sym.FullyConnected(data=pred, num_hidden=1000, name='pred')
    pred = mx.sym.SoftmaxOutput(data=pred, name='softmax')
    
    mod = mx.mod.Module(symbol=pred, 
                    context=mx.gpu(0),
                    data_names=['text','image'],
                    #data_names=['data1'],
                    label_names = ['softmax_label']
                    )
    
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

    mod.bind(data_shapes=data_shapes, label_shapes = label_shapes)
    mod.init_params()
    mod.fit(data_train, data_eva, num_epoch=400, eval_metric=eval_metrics(),
            #batch_end_callback=mx.callback.Speedometer(batch_size,20),
            epoch_end_callback= evaluation_callback)


if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    
    train(args)
    
