import matplotlib.pyplot as plt
import numpy as np

infile = r"/home/ec2-user/workplace/VQA/concat.log"

train_acc = []
train_acc_key = ["Train-accuracy"]
val_acc = []
val_acc_key = ["Validation-accuracy"]
train_ce = []
train_ce_key = ["Train-cross-entropy"]
val_ce = []
val_ce_key = ["Validation-cross-entropy"]


with open(infile) as f:
    f = f.readlines()

for line in f:
    for phrase in train_acc_key:
        if phrase in line:
            idx = line.index('=')
            train_acc.append(line[idx+1:len(line)-1])
    for phrase in val_acc_key:
        if phrase in line:
            idx = line.index('=')
            val_acc.append(line[idx+1:len(line)-1])
    for phrase in train_ce_key:
        if phrase in line:
            idx = line.index('=')
            train_ce.append(line[idx+1:len(line)-1])
    for phrase in val_ce_key:
        if phrase in line:
            idx = line.index('=')
            val_ce.append(line[idx+1:len(line)-1])

fig, ax1 = plt.subplots()
#t = np.arange(0,len(train_acc),1)
ax1.plot(train_acc, 'b',linestyle = '--', label = 'train acc')
ax1.plot(val_acc,'b',label = 'val acc')
ax1.set_xlabel('Epoches')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('acc', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(train_ce, 'r', linestyle = '--', label = 'train ce')
ax2.plot(val_ce,'r', label = 'val ce')
ax2.set_ylabel('ce', color='r')
ax2.tick_params('y', colors='r')
ax1.legend(loc=2, fancybox=True, framealpha=0.5)
ax2.legend(loc=1, fancybox=True, framealpha=0.5)

fig.tight_layout()
plt.show()
