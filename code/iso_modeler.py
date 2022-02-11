#Import libraries
from os.path import join
import pandas as pd
from isocompy.data_preparation import preprocess
from isocompy.reg_model import model
from isocompy.tools import session, evaluation, stats, plots



#d=join("case_studies","salar_de_atacama","all_data_auto_filtered")
#d=join("case_studies","salar_de_atacama","all_data_no_filter")
d=join("case_studies","salar_de_atacama","manual_filtered_data_v4")

dir_inp=join(d,"inputs")
dir_outs=join(d,"outputs")

#load desired isotope class:
pkldir=r"case_studies/salar_de_atacama/manual_filtered_data_v4/outputs/st1_all_month/est_class_st1_18_Jan_2022_20_45_st1True_st2True.pkl"
est_class=session.load(pkldir)
#-------------------------------------------

#stage 2 model: The predicted features from st2 will be used:
#determine the input and output of the second stage model:

st2_model_var_dict={"iso_18":["CooX","CooY","CooZ","tmp","prc","hmd"],"iso_2h":["CooX","CooY","CooZ","tmp","prc","hmd"]}

args_dic={"feature_selection":"auto","vif_threshold":5, "vif_selection_pairs":[["CooZ","tmp"]], "correlation_threshold":0.87,"vif_corr":True,"p_val":0.05}

est_class.st2_fit(model_var_dict=st2_model_var_dict,args_dic=args_dic)
#save stage 2
est_class_st2_dir=session.save(est_class,name='est_class_st2') 

#stage 2 model plots
plots.best_estimator_plots(est_class,st1=False)
plots.partial_dep_plots(est_class,st1=False) 
#-------------------------------------------
