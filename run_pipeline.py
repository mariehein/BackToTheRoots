import numpy as np
import dataprep_utils as dp
import plotting_utils as pf
import argparse
import os

parser = argparse.ArgumentParser(
    description='Run the full analysis chain.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#For SIC curve reproduction only these 5 option need to be changed
parser.add_argument('--mode', type=str, choices=["IAD", "supervised"], required=True)
parser.add_argument('--directory', type=str, required=True)
parser.add_argument('--classifier', type=str, choices=["BDT", "NN"], required=True)
parser.add_argument('--input_set', type=str, choices=["baseline","extended1","extended2","extended3","kitchensink"])
parser.add_argument('--gaussian_inputs', type=int, default=None)

#Data files
parser.add_argument('--data_file', type=str, default="/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_v2.extratau_2.features.h5")
parser.add_argument('--extrabkg_file', type=str, default="/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_DelphesPythia8_v2_qcd_extra_inneronly_combined_extratau_2_features.h5")
parser.add_argument('--signal_file', type=str, default=None)
parser.add_argument('--three_pronged', type=False, action="store_true")

#Dataset preparation
parser.add_argument('--signal_percentage', type=float, default=None, description="Third in priority order")
parser.add_argument('--signal_number', type=int, default=None, description="Second in priority order")
parser.add_argument('--minmass', type=float, default=3.3)
parser.add_argument('--maxmass', type=float, default=3.7)
parser.add_argument('--cl_logit', default=False, action="store_true")
parser.add_argument('--cl_norm', default=True, action="store_false")
parser.add_argument('--gaussian_inputs', default=False, action="store_true")
parser.add_argument('--N_normal_inputs', default=4, type=int, description="Needed only for gaussian inputs")
parser.add_argument('--supervised_normal_signal', default=False, action="store_true")
parser.add_argument('--set_seed', type=int, default=1)
parser.add_argument('--randomize_seed', default=False, action="store_true")
parser.add_argument('--inputs', type=int, default=4)

#2D scan
parser.add_argument('--scan_2D', default=False, action="store_true")
parser.add_argument('--N_CR', type=int, default=None)
parser.add_argument('--N_bkg', type=int, default=None)
parser.add_argument('--signal_significance', type=float, default=None, description="Takes highest priority, only supported for 2D_scan")

#General classifier Arguments
parser.add_argument('--N_runs', type=int, default=10)
parser.add_argument('--ensemble_over', default=50, type=int)
parser.add_argument('--start_at_run', type=int, default=0)
parser.add_argument('--cl_N_best_epochs', type=int, default=1)
parser.add_argument('--cl_filename', type=str, default="classifier4.yml")

args = parser.parse_args()

if not os.path.exists(args.directory):
	os.makedirs(args.directory)

if args.classifier=="NN":
    args.N_runs=100
    args.cl_N_best_epochs=1
    import NN_utils as NN
else:
    import BDT_utils as BDT

if args.input_set=="extended1":
    args.inputs=10
elif args.input_set=="extended2":
    args.inputs=12
elif args.input_set=="extended3":
    args.inputs=56
elif args.input_set=="kitchensink":
    args.inputs=72

if args.three_pronged:
	args.signal_file = "/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_Z_XY_qqq.extratau_2.features.h5"

if args.gaussian_inputs is not None:
    args.inputs+=args.gaussian_inputs

if args.signal_significance is not None:
    if args.scan_2D:
        args.signal_number = int(args.signal_significance*np.sqrt(args.N_bkg))
        args.randomize_seed = True
    else: 
        raise ValueError("signal significance only supported for 2D_scan")

if not args.scan_2D:
    if not args.randomize_seed:
        X_train, Y_train, X_test, Y_test = dp.classifier_data_prep(args)
    for i in range(args.start_at_run, args.N_runs):
        print()
        print("------------------------------------------------------")
        print()
        print("Classifier run no. ", i)
        print()
        args.set_seed = i
        if args.classifier=="NN": 
            direc_run = args.directory+"run"+str(i)+"/"
            if args.randomize_seed and i%args.ensemble_over==0:
                X_train, Y_train, X_test, Y_test = dp.classifier_data_prep(args)
            NN.classifier_training(args, X_train, Y_train, X_test, Y_test, i, direc_run=direc_run)
        else:
            if args.randomize_seed:
                X_train, Y_train, X_test, Y_test = dp.classifier_data_prep(args)
            BDT.classifier_training(args, X_train, Y_train, X_test, Y_test, i)

else: 
    if args.N_bkg is None:
        raise ValueError("Need to specify N_bkg")
    if args.classifier=="NN":
        raise ValueError("NN 2D scan currently not supported.")
    else:
        for i in range(args.start_at_run, args.N_runs):
            print()
            print("------------------------------------------------------")
            print()
            print("Classifier run no. ", i)
            print()
            args.set_seed = i
            if args.randomize_seed:
                X_train, Y_train, X_test, Y_test = dp.data_prep_2D(args)
            BDT.classifier_training(args, X_train, Y_train, X_test, Y_test, i)
