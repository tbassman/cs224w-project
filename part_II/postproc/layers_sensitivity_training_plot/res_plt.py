import matplotlib.pyplot as plt
import numpy as np

# read logs
files = ['./prn_cwq-rearev-sbert_gnn3.log', \
         './prn_cwq-rearev-sbert_gnn5.log', \
         './prn_cwq-rearev-sbert_gnn7.log' \
        ]

plt_res = {}
plt_res1 = {}
for file in files:
    tmp_r = []
    tmp_l = []
    with open(file, 'r') as f:
        lines = f.readlines()

        #for i, line in enumerate(lines):
        cnt = 0
        while cnt < len(lines):
            if not 'Epoch:' in lines[cnt]:
                cnt += 1
                continue

            tmp_cnt = lines[cnt: cnt + 6]
            #tmp = tmp_cnt[-1].strip().split('TEST ')[1].split(',')

            tmp_r.append(tmp_cnt[-1].strip().split('TEST ')[-1])

            tmp_l.append(float(tmp_cnt[0].strip().split('loss ')[-1].split(" ")[1][:-1]))
            tmp_l.append(float(tmp_cnt[2].strip().split('loss ')[-1].split(" ")[1][:-1]))

            cnt += 6

    plt_res[file.split('.')[1].split('_')[-1]] = tmp_r
    plt_res1[file.split('.')[1].split('_')[-1]] = tmp_l

def fetch_data(metric, data_in):
    indx = None
    if metric == 'F1': indx = 0
    elif metric == 'H1': indx = 1
    elif metric == 'EM': indx = 2
    else: raise Exception("Not support metric")

    res = []
    for elm in data_in:
        res.append(float(elm.split(',')[indx].split(' ')[-1]))
    return res

#process data
res_f1 = {}
for k, v in plt_res.items():
    res_f1[k] = fetch_data('F1', v)

res_h1 = {}
for k, v in plt_res.items():
    res_h1[k] = fetch_data('H1', v)


# Create a figure and an axes object
fig, ax = plt.subplots()

# Plot the data
labels = ["gnn L=3", "gnn L=5", "gnn L=7"]
x = np.arange(0, 100, 2, dtype=int)
y = res_f1['gnn3']
#ax.plot(x, y, label=labels[0])
line0, = plt.plot(x, y, label=labels[0])

y = res_f1['gnn5']
#ax.plot(x, y, label=labels[1])
line1, = plt.plot(x, y, label=labels[1])

y = res_f1['gnn7']
#ax.plot(x, y, label=labels[1])
line2, = plt.plot(x, y, label=labels[2])

# Add labels and title
ax.set_xlabel("Epoch")
ax.set_ylabel("F1 Score")
ax.set_title("F1 Score")
#plt.rcParams["figure.figsize"] = [7.50, 3.50]
#plt.rcParams["figure.autolayout"] = True

leg = plt.legend(loc='lower right')
plt.savefig("F1.png",
            #bbox_inches ="tight",
            #pad_inches = 1,
            #transparent = True,
            #facecolor ="g",
            #edgecolor ='w',
            #orientation ='landscape'
            )
#plt.show()
plt.close(fig)

# Create a figure and an axes object
fig, ax = plt.subplots()

# Plot the data
labels = ["gnn L=3", "gnn L=5", "gnn L=7"]
x = np.arange(0, 100, 2, dtype=int)
y = res_h1['gnn3']
#ax.plot(x, y, label=labels[0])
line0, = plt.plot(x, y, label=labels[0])

y = res_h1['gnn5']
#ax.plot(x, y, label=labels[1])
line1, = plt.plot(x, y, label=labels[1])

y = res_h1['gnn7']
#ax.plot(x, y, label=labels[1])
line2, = plt.plot(x, y, label=labels[2])

# Add labels and title
ax.set_xlabel("Epoch")
ax.set_ylabel("H1 Score")
ax.set_title("H1 Score")
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

leg = plt.legend(loc='lower right')
plt.savefig("H1.png",
            #bbox_inches ="tight",
            #pad_inches = 1,
            #transparent = True,
            #facecolor ="g",
            #edgecolor ='w',
            #orientation ='landscape'
            )
#plt.show()
plt.close(fig)

# Create a figure and an axes object
fig, ax = plt.subplots()

# Plot the data
labels = ["gnn L=3", "gnn L=5", "gnn L=7"]
x = np.arange(0, 100, 1, dtype=int)
y = plt_res1['gnn3']
#plt.yscale('log')
#ax.plot(x, y, label=labels[0])
line0, = plt.plot(x, y, label=labels[0])

y = plt_res1['gnn5']
#ax.plot(x, y, label=labels[1])
line1, = plt.plot(x, y, label=labels[1])

y = plt_res1['gnn7']
#ax.plot(x, y, label=labels[1])
line2, = plt.plot(x, y, label=labels[2])

# Add labels and title
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Average KL Loss")
#plt.rcParams["figure.figsize"] = [7.50, 3.50]
#plt.rcParams["figure.autolayout"] = True


leg = plt.legend(loc='upper right')
plt.savefig("avg_kl.png",
            #bbox_inches ="tight",
            #pad_inches = 1,
            #transparent = True,
            #facecolor ="g",
            #edgecolor ='w',
            #orientation ='landscape'
            )
#plt.show()
plt.close(fig)

tmp = 0