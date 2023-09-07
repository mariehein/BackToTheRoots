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
parser.add_argument('--three_prong', type=False, action="store_true")

#Dataset preparation
parser.add_argument('--signal_percentage', type=float, default=None)
parser.add_argument('--signal_significance', type=float, default=None)
parser.add_argument('--signal_number', type=int, default=None)
parser.add_argument('--minmass', type=float, default=3.3)
parser.add_argument('--maxmass', type=float, default=3.7)
parser.add_argument('--cl_logit', default=False, action="store_true")
parser.add_argument('--cl_norm', default=True, action="store_false")
parser.add_argument('--gaussian_inputs', default=False, action="store_true")
parser.add_argument('--N_normal_inputs', default=4, type=int, description="Needed only for gaussian inputs")
parser.add_argument('--supervised_normal_signal', default=False, action="store_true")
parser.add_argument('--set_seed', type=int, default=1)
parser.add_argument('--randomize_seed', type=False, action="store_true")
parser.add_argument('--inputs', type=int, default=4)

#General classifier Arguments
parser.add_argument('--use_class_weights', default=True, action="store_false")
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

if args.gaussian_inputs is not None:
    args.inputs+=args.gaussian_inputs

if args.classifier=="NN":
