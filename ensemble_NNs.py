import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.cm as col
import seaborn as sns
import argparse
import warnings
import os
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--directory', type=str, required=True)
parser.add_argument('--scan_1D', default=False, action="store_true")
args = parser.parse_args()

max_err=0.2

S = 20000
B = 312858

plt.rcParams['pgf.rcfonts'] = False
plt.rcParams['font.serif'] = []
#plt.rcParams['text.usetex'] = True
#plt.rcParams['figure.figsize'] = 3.5, 2.625
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['errorbar.capsize'] = 2
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.title_fontsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.frameon'] = False

N_avg = [10, 50, None]

def make_one_array(twod_arr, new_arr):
    if len(new_arr) < len(twod_arr.T):
        app = np.zeros(len(twod_arr.T)-len(new_arr),dtype=None)
        new_arr = np.concatenate((new_arr, app), axis=0)
    elif len(twod_arr.T) < len(new_arr):
        app = np.zeros((len(twod_arr), len(new_arr)-len(twod_arr.T)), dtype=None)
        twod_arr = np.concatenate((twod_arr, app), axis=1)
    return np.concatenate((twod_arr, np.array([new_arr])),axis=0)

def make_ROCs(folder, Y_test=None):
    length=0
    while os.path.exists(folder+"run"+str(length)+"/"):
        length+=1
    #length-=1
    print(length)
    
    if Y_test is None:
        Y_test = np.load(folder+"Y_test.npy")
    preds = np.zeros((length, len(Y_test)))
    for i in range(length):
        preds[i] = np.load(folder+"run"+str(i)+"/preds.npy")
    
    for e in N_avg:
        if e is None: 
            e = length
        for i in range(int(length/e)):
            results = np.mean(preds[i*e:(i+1)*e], axis=0)
            fpr, tpr, _ = roc_curve(Y_test[:,1], results)
            #print(roc_auc_score(Y_test[:,1], results))
            if i==0:
                fpr_arr = np.reshape(fpr, (1, len(fpr)))
                tpr_arr = np.reshape(tpr, (1, len(tpr)))
            else:
                fpr_arr = make_one_array(fpr_arr, fpr)
                tpr_arr = make_one_array(tpr_arr, tpr)
        np.save(folder+"fpr_"+str(e)+"_temp.npy", fpr_arr)
        np.save(folder+"tpr_"+str(e)+"_temp.npy", tpr_arr)

if args.scan_1D:
    for i in [0,100, 200, 300, 400, 500, 750, 1000, 1200, 1500, 2000]:
        make_ROCs(args.directory + "Nsig_"+str(i)+"/")
else:
    make_ROCs(args.directory)

