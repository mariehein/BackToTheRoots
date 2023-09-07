import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from pathlib import Path
import warnings

def plot_training(history,title="training", directory=None):
	plt.figure()
	plt.plot(history.history["loss"])
	plt.title(title)
	plt.ylabel("loss")
	plt.xlabel("epoch")
	if directory is None:
		plt.savefig("plots/training/"+title+"_train.pdf")
	else:
		plt.savefig(directory+"training_train.pdf")
	plt.show()
    
	plt.figure()
	plt.plot(history.history["val_loss"])
	plt.title(title)
	plt.ylabel("val loss")
	plt.xlabel("epoch")
	if directory is None:
		plt.savefig("plots/training/"+title+"_val.pdf")
	else:
		plt.savefig(directory+"training_val.pdf")
	plt.show()
	return

def make_one_array(twod_arr,new_arr):
	if len(new_arr) < len(twod_arr.T):
		app=np.zeros(len(twod_arr.T)-len(new_arr),dtype=None)
		new_arr=np.concatenate((new_arr,app),axis=0)
	elif len(twod_arr.T) < len(new_arr):
		app=np.zeros((len(twod_arr),len(new_arr)-len(twod_arr.T)),dtype=None)
		twod_arr=np.concatenate((twod_arr,app),axis=1)
	return np.concatenate((twod_arr,np.array([new_arr])),axis=0)

def plot_roc(test_results, test_labels, title="roc", directory = None, direc_run = None, plot=False, save_AUC=False):
    if direc_run is None:
        direc_run=directory
        
    fpr, tpr, _ = roc_curve(test_labels, test_results)
    auc = roc_auc_score(test_labels, test_results)

    if plot:
        x = np.linspace(0.001, 1, 10000)
        plt.figure()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
            plt.plot(tpr, 1 / fpr, label="model")
        plt.plot(x, 1 / x, color="black", linestyle="--", label="random")
        plt.legend()
        plt.grid()
        plt.yscale("log")
        plt.ylim(1, 1e5)
        plt.xlim(0,1)
        plt.ylabel(r"1/$\epsilon_B$")
        plt.xlabel(r"$\epsilon_S$")
        plt.title(title)
        plt.savefig(direc_run+title+"roc.pdf")

    if Path(directory+"tpr_"+title+".npy").is_file():
        tpr_arr=np.load(directory+"tpr_"+title+".npy")
        np.save(directory+"tpr_"+title+".npy",make_one_array(tpr_arr,tpr))
        fpr_arr=np.load(directory+"fpr_"+title+".npy")
        np.save(directory+"fpr_"+title+".npy",make_one_array(fpr_arr,fpr))
    else: 
        np.save(directory+"tpr_"+title+".npy",np.array([tpr]))
        np.save(directory+"fpr_"+title+".npy",np.array([fpr]))

    if save_AUC:	
        f=open(directory+title+".txt",'a+')
        f.write("\n"+str(auc))
    return auc

def save_loss(score, title, directory, direc_run = None):
	if direc_run is None:
		direc_run=directory

	if Path(directory+title+".npy").is_file():
		val_loss_arr=np.load(directory+title+".npy")
		np.save(directory+title+".npy",np.append(val_loss_arr,score))
	else: 
		np.save(directory+title+".npy",np.array([score]))
	
def plot_scores(results, labels, title="scores", name0="background", name1="signal", directory=None):
	plt.figure()
	plt.title(title)
	plt.hist([results[labels==1],results[labels==0]],50,label=[name1,name0],density=True)
	plt.xlabel("classifier score")
	plt.legend()
	if directory is None:
		plt.savefig("plots/scores/"+title+".pdf")
	else:
		plt.savefig(directory+"scores"+title+".pdf")
	return 