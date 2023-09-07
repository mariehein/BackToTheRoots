import numpy as np
import plotting_utils as pf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss



def classifier_training(X_train, Y_train, X_test, Y_test, args, run):

    class_weight = {0: 1, 1: len(Y_train)/sum(Y_train.T[1])-1}
    class_weights = class_weight[0]*Y_train[:,0]+class_weight[1]*Y_train[:,1]

    print("\nTraining class weights: ", class_weight)

    test_results = np.zeros((args.ensemble_over,len(X_test)))

    for j in range(args.ensemble_over):
        print("Tree number:", args.ensemble_over*run+j)
        np.random.seed(run*args.ensemble_over+j+1)
        tree = HistGradientBoostingClassifier(verbose=1, max_iter=200, max_leaf_nodes=31, validation_fraction= 0.5)
        results_f = tree.fit(X_train, Y_train[:,1], sample_weight=class_weights)
        test_results[j] = tree.predict_proba(X_test)[:,1]
        print("AUC last epoch: %.3f" % pf.plot_roc(test_results[j], Y_test[:,1],title="roc_classifier",directory=args.directory))
        del tree
        del results_f
    test_results = np.mean(test_results, axis=0)
    print("AUC last epoch: %.3f" % pf.plot_roc(test_results, Y_test[:,1],title="roc_classifier_averaging",directory=args.directory))
