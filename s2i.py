import json

print('loading json file...')

with open('test_result.json') as data_file:
    data = json.load(data_file)

for i in xrange(0, 121512):
    print i
    data[i]['question_id'] = int(data[i]['question_id'])

dd = json.dump(data,open('/home/ec2-user/workplace/vqa-eva/VQA/Results/OpenEnded_mscoco_lstm_results.json','w'))
