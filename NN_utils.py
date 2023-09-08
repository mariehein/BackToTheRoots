from tensorflow import keras
import numpy as np
import os
import plotting_utils as pf
import yaml

models = keras.models
layers = keras.layers
regularizers = keras.regularizers

def make_model(activation="relu",hidden=3,inputs=4,lr=1e-3,dropout=0.1, l1=0, l2 =0, momentum = 0.9, label_smoothing=0):
	model = models.Sequential()
	model.add(layers.Dense(64,input_shape=(inputs,)))
	for i in range(hidden-1):
		if activation =="relu":
			model.add(layers.ReLU())
		elif activation == "leaky":
			model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.Dropout(dropout))
		model.add(layers.Dense(64,kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
	model.add(layers.Dense(2, activation="softmax"))

	loss = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

	model.compile(
		loss=loss,
		optimizer=keras.optimizers.Adam(lr, beta_1=momentum),
		metrics=["accuracy"],
		weighted_metrics=[],
	)
	return model

def classifier_training(X_train, Y_train, X_test, Y_test, args, run, direc_run=None):
	if direc_run is None:	
		direc_run=args.directory
	
	with open(args.cl_filename, 'r') as stream:
		params = yaml.safe_load(stream)

	model = make_model(activation=params['activation'], hidden=int(params['hidden']), inputs=args.inputs, lr=float(params['lr']), dropout=float(params['dropout']), l1=float(params['l1']), l2 =float(params['l2']), momentum = float(params['beta_1']), label_smoothing=float(params['label_smoothing']))

	if not os.path.exists(direc_run):
		os.makedirs(direc_run)

	earlystopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
	callbacks = [earlystopping]

	np.random.seed(run)
	inds = np.array(range(len(X_train)))
	np.random.shuffle(inds)
	X_train, X_val = np.array_split(X_train,2)
	Y_train, Y_val = np.array_split(Y_train,2)

	class_weight = {0: 1, 1: len(Y_train)/sum(Y_train.T[1])-1}
	val_weight = {0: 1, 1: len(Y_val)/sum(Y_val.T[1])-1}
	val_sample_weights = val_weight[0]*Y_val[:,0]+val_weight[1]*Y_val[:,1]
	
	results = model.fit(
		X_train,
		Y_train,
		batch_size=params['batchsize'],
		epochs=params['epochs'],
		shuffle=True,
		verbose=2,
		validation_data=(X_val, Y_val, val_sample_weights),
		class_weight=class_weight,
		callbacks=callbacks,
	)

	np.save(direc_run+'classifier_history.npy', results.history)
	
	test_results = model.predict(X_test, verbose=0).T[1]
	print("AUC with averaging: %.3f" % pf.plot_roc(test_results, Y_test[:,1], title="roc_NN",directory=args.directory, direc_run=direc_run))
	np.save(direc_run+"preds.npy", test_results)
	
	return model, results