import numpy as np
from scipy.special import logit, expit
import pandas as pd
import warnings

def shuffle_XY(X,Y):
    seed_int=np.random.randint(300)
    np.random.seed(seed_int)
    np.random.shuffle(X)
    np.random.seed(seed_int)
    np.random.shuffle(Y)
    return X,Y

def to_categorical(Y, N_classes):
	Y=np.array(Y,dtype=int)
	return np.eye(N_classes)[Y]

class no_logit_norm:
	def __init__(self,array):
		self.mean = np.mean(array, axis=0)
		self.std = np.std(array, axis=0)

	def forward(self,array0):
		return (np.copy(array0)-self.mean)/self.std, np.ones(len(array0),dtype=bool)

	def inverse(self,array0):
		return np.copy(array0)*self.std+self.mean

class logit_norm:
    def __init__(self, array0, mean=True):
        array = np.copy(array0)
        self.shift = np.min(array, axis=0)
        self.num = len(self.shift)
        self.max = np.max(array, axis=0) + self.shift
        if mean:
            finite=np.ones(len(array),dtype=bool)
            for i in range(self.num):
                array[:, i] = logit((array[:, i] + self.shift[i]) / self.max[i])
                finite *= np.isfinite(array[:, i])
            array=array[finite]
            self.mean = np.nanmean(array,axis=0)
            print(self.mean)
            self.std = np.nanstd(array,axis=0)
            print(self.std)
        self.do_mean = mean

    def forward(self, array0):
        array = np.copy(array0)
        finite = np.ones(len(array), dtype=bool)
        for i in range(self.num):
            array[:, i] = logit((array[:, i] + self.shift[i]) / self.max[i])
            if self.do_mean:
                array[:,i] = (array[:,i]-self.mean[i])/self.std[i]
            finite *= np.isfinite(array[:, i])
        return array[finite, :], finite

    def inverse(self, array0):
        array = np.copy(array0)
        for i in range(self.num):
            if self.do_mean:
                array[:,i] = array[:,i]*self.std[i] + self.mean[i]
            array[:,i] =expit(array[:,i])
            array[:, i] = array[:, i] * self.max[i] - self.shift[i]
        return array

def make_features_baseline(features, label_arr):
    E_part = np.sqrt(features[:,0]**2+features[:,1]**2+features[:,2]**2+features[:,3]**2)+np.sqrt(features[:,7]**2+features[:,8]**2+features[:,9]**2+features[:,10]**2)
    p_part2 = (features[:,0]+features[:,7])**2+(features[:,1]+features[:,8])**2+(features[:,2]+features[:,9])**2
    m_jj = np.sqrt(E_part**2-p_part2)
    ind=np.array(features[:,10]> features[:,3]).astype(int)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
        feat1 = np.array([m_jj*1e-3, features[:, 3]*1e-3, (features[:,10]-features[:, 3])*1e-3, features[:, 5]/features[:,4], features[:, 12]/features[:,11], features[:, 6]/features[:,5], features[:, 13]/features[:,12], label_arr])
        feat2 = np.array([m_jj*1e-3, features[:, 10]*1e-3, (features[:,3]-features[:, 10])*1e-3, features[:, 12]/features[:,11], features[:, 5]/features[:,4], features[:, 13]/features[:,12], features[:, 6]/features[:,5], label_arr])
    return np.nan_to_num(feat1*ind+feat2*(np.ones(len(ind))-ind))

def make_features_extended12(features_j1, features_j2, label_arr, set):
	E_part2 = np.sqrt(features_j1[:,0]**2+features_j1[:,1]**2+features_j1[:,2]**2+features_j1[:,3]**2)+np.sqrt(features_j2[:,0]**2+features_j2[:,1]**2+features_j2[:,2]**2+features_j2[:,3]**2)
	p_part2 = (features_j1[:,0]+features_j2[:,0])**2 + (features_j1[:,1]+features_j2[:,1])**2 + (features_j1[:,2]+features_j2[:,2])**2
	m_jj = np.sqrt(E_part2**2-p_part2)

	ind_not = np.argwhere(features_j1[:,3]> features_j2[:,3])
	f_j1 = np.copy(features_j1)
	features_j1[ind_not] = features_j2[ind_not]
	features_j2[ind_not] = f_j1[ind_not]
	del f_j1	

	if set=="extended2":
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
			warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
			features = np.zeros((len(m_jj), 14))
			features[:,0] = m_jj * 1e-3
			features[:,1] = features_j1[:, 3] * 1e-3
			features[:,2] = (features_j2[:, 3] - features_j1[:, 3]) *1e-3
			for i in range(5):
				features[:,3+2*i] = features_j1[:,4+i]
				features[:,3+2*i+1] = features_j2[:,4+i]
			features[:,-1] = label_arr
	elif set=="extended1":
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
			warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
			inputs = 12
			if add_pt:
				inputs+=2
			features = np.zeros((len(m_jj), inputs))
			features[:,0] = m_jj * 1e-3
			features[:,1] = features_j1[:, 3] * 1e-3
			features[:,2] = (features_j2[:, 3] - features_j1[:, 3]) *1e-3
			for i in range(4):
				features[:,3+2*i] = features_j1[:,5+i]/features_j1[:,4+i]
				features[:,3+2*i+1] = features_j2[:,5+i]/features_j2[:,4+i]
			if add_pt:
				features[:,-3] = np.sqrt(features_j1[:,0]**2+features_j1[:,1]**2)
				features[:,-2] = np.sqrt(features_j2[:,0]**2+features_j2[:,1]**2)
			features[:,-1] = label_arr

	return np.nan_to_num(features)

def make_features_extended3(pandas_file, label_arr, set):
    features_j1 = np.array(pandas_file[['pxj1', 'pyj1', 'pzj1', 'mj1']], dtype=np.float32)
    features_j2 = np.array(pandas_file[['pxj2', 'pyj2', 'pzj2', 'mj2']], dtype=np.float32)
    
    E_part2 = np.sqrt(features_j1[:,0]**2+features_j1[:,1]**2+features_j1[:,2]**2+features_j1[:,3]**2)+np.sqrt(features_j2[:,0]**2+features_j2[:,1]**2+features_j2[:,2]**2+features_j2[:,3]**2)
    p_part2 = (features_j1[:,0]+features_j2[:,0])**2 + (features_j1[:,1]+features_j2[:,1])**2 + (features_j1[:,2]+features_j2[:,2])**2
    m_jj = np.sqrt(E_part2**2-p_part2)

    beta = [5, 1, 2]
    jet = ["j1", "j2"]
    to_subjettiness=9
    subjettinesses = np.zeros((len(m_jj),2,to_subjettiness*3))
    for k,b in enumerate(beta):
        for l, j in enumerate(jet):
            names = ["tau"+str(i)+j+"_"+str(b) for i in range(1,to_subjettiness+1)]
            subjettinesses[:,l,k*to_subjettiness:(k+1)*to_subjettiness]=np.array(pandas_file[names], dtype=np.float32)

    if set=="kitchensink":
        ratios = np.zeros((len(m_jj), 2, to_subjettiness-1))
        for l, j in enumerate(jet):
            names = ["tau"+str(i)+j+"_1" for i in range(1,to_subjettiness+1)]
            for i in range(len(names)-1):
                ratios[:,l, i]=np.array(pandas_file[names[i+1]], dtype=np.float32)/np.array(pandas_file[names[i]], dtype=np.float32)

    ind_not = np.argwhere(features_j1[:,3]> features_j2[:,3])
    f_j1 = np.copy(features_j1)
    features_j1[ind_not] = features_j2[ind_not]
    features_j2[ind_not] = f_j1[ind_not]
    del f_j1
    s = np.copy(subjettinesses)
    subjettinesses[ind_not,0] = subjettinesses[ind_not,1] 
    subjettinesses[ind_not,1] = s[ind_not,0] 
    del s
    if set=="kitchen_sink":
        r = np.copy(ratios)
        ratios[ind_not,0] = ratios[ind_not,1] 
        ratios[ind_not,1] = r[ind_not,0] 
        del r

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
        warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
        features = np.zeros((len(m_jj), 4+to_subjettiness*6))
        if set=="kitchensink":
            features = np.zeros((len(m_jj), 4+to_subjettiness*8-2))
        features[:,0] = m_jj * 1e-3
        features[:,1] = features_j1[:, 3] * 1e-3
        features[:,2] = (features_j2[:, 3] - features_j1[:, 3]) *1e-3
        features[:,3:3+subjettinesses.shape[-1]]= subjettinesses[:,0]
        features[:,3+subjettinesses.shape[-1]:3+subjettinesses.shape[-1]*2]= subjettinesses[:,1]
        if set=="kitchensink":
            features[:,3+subjettinesses.shape[-1]*2:3+subjettinesses.shape[-1]*2+(to_subjettiness-1)] = ratios[:,0] 
            features[:,3+subjettinesses.shape[-1]*2+(to_subjettiness-1) : 3+subjettinesses.shape[-1]*2+(to_subjettiness-1)*2] = ratios[:,1] 
        features[:,-1] = label_arr

    return np.nan_to_num(features)

def file_loading(filename, args, labels=True, signal=0):
    pandas_file = pd.read_hdf(filename)
    if labels:
        label_arr = pandas_file['labels']
    else: 
        label_arr = np.ones((len(features)))*signal

    if args.input_set == "baseline":
        features = pandas_file['pxj1', 'pyj1', 'pzj1', 'mj1', 'tau1j1_1', 'tau2j1_1', 'tau3j1_1', 'pxj2', 'pyj2', 'pzj2', 'mj2', 'tau1j2_1', 'tau2j2_1', 'tau3j2_1']
        features = make_features_baseline(features, label_arr)
    elif args.input_set in ["extended1", "extended2"]:
        features_j1 = np.array(pd.read_hdf(filename)[['pxj1', 'pyj1', 'pzj1', 'mj1', 'tau1j1_1', 'tau2j1_1', 'tau3j1_1', 'tau4j1_1', 'tau5j1_1']])
        features_j2 = np.array(pd.read_hdf(filename)[['pxj2', 'pyj2', 'pzj2', 'mj2', 'tau1j2_1', 'tau2j2_1', 'tau3j2_1', 'tau4j2_1', 'tau5j2_1']])
        features = make_features_extended12(features_j1, features_j2, label_arr, args.input_set)
        del features_j1, features_j2
    elif args.input_set in ["extended3","kitchensink"]:
        features = make_features_extended3(pandas_file, label_arr, args.input_set)
    pandas_file.close()
    return features

def classifier_data_prep(args, samples=None):
    data = file_loading(args.data_file, args)
    extra_bkg = file_loading(args.extrabkg_file, args, labels=False)
    if args.signal_file is not None: 
        data_signal = file_loading(args.signal_file, args, labels=False, signal=1)

    if args.signal_file is not None: 
        sig = data_signal
    else:
        sig = data[data[:,-1]==1]
    bkg = data[data[:,-1]==0]

    # ADD SIGNAL OPTIONS
    if args.signal_percentage is None:
        n_sig = 1000
    else:
        n_sig = int(args.signal_percentage*1000/0.6361658645922605)
    print("n_sig=", n_sig)

    data_all = np.concatenate((bkg,sig[:n_sig]),axis=0)
    np.random.seed(args.set_seed)
    np.random.shuffle(data_all)
    extra_sig = sig[n_sig:]
    innersig_mask = (extra_sig[:,0]>args.minmass) & (extra_sig[:,0]<args.maxmass)
    inner_extra_sig = extra_sig[innersig_mask]

    innermask = (data_all[:,0]>args.minmass) & (data_all[:,0]<args.maxmass)
    innerdata = data_all[innermask]

    extrabkg1 = extra_bkg[:312858]
    extrabkg2 = extra_bkg[312858:]

    samples_train = extrabkg1[40000:]

    if args.mode=="supervised":
        if not args.supervised_normal_signal:
            sig_train = inner_extra_sig[20000:]
        else: 
            sig_train = innerdata[:120000]
            sig_train = innerdata[:60000]
        X_train = np.concatenate((samples_train, sig_train), axis=0)
        Y_train = X_train[:,-1]
        if args.gaussian_inputs:
            gauss = np.random.normal(size=(len(X_train),args.inputs-args.N_normal_inputs))
            X_train = np.concatenate((X_train[:,1:args.N_normal_inputs+1],gauss), axis=1)
        else:
            X_train = X_train[:,1:args.inputs+1]
    elif args.mode=="IAD":
        if args.gaussian_inputs:
            X_train = np.concatenate((innerdata[:60000,1:args.N_normal_inputs+1],samples_train[:,1:args.N_normal_inputs+1]),axis=0)
            gauss = np.random.normal(size=(len(X_train),args.inputs-args.N_normal_inputs))
            X_train = np.concatenate((X_train,gauss), axis=1)
            print(X_train.shape)
        else:
            X_train = np.concatenate((innerdata[:60000,1:args.inputs+1],samples_train[:,1:args.inputs+1]),axis=0)
        Y_train = np.concatenate((np.ones(len(X_train)-len(samples_train)),np.zeros(len(samples_train))),axis=0)		
    else: 
        raise ValueError('Wrong --args.mode given')

    X_test = np.concatenate((extrabkg2,inner_extra_sig[:20000],extrabkg1[:40000]))
    Y_test = to_categorical(X_test[:,-1],2)
    if args.gaussian_inputs:
        gauss = np.random.normal(size=(len(X_test),args.inputs-args.N_normal_inputs))
        X_test = np.concatenate((X_test[:,1:args.N_normal_inputs+1],gauss), axis=1)
    else:
        X_test = X_test[:,1:args.inputs+1]

    X_train, Y_train = shuffle_XY(X_train, to_categorical(Y_train,2))

    if args.cl_norm:
        if args.cl_logit:
            normalisation = logit_norm(X_train)
        else:
            normalisation = no_logit_norm(X_train)
        X_train, _ = normalisation.forward(X_train)
        X_test, _ = normalisation.forward(X_test)

    print("Train set: ", len(X_train), "; Test set: ", len(X_test))

    np.save(args.directory+"X_train.npy", X_train)
    np.save(args.directory+"X_test.npy", X_test)
    np.save(args.directory+"Y_train.npy", Y_train)
    np.save(args.directory+"Y_test.npy", Y_test)

    return X_train, Y_train, X_test, Y_test