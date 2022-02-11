#Import libraries
from os.path import join
import pandas as pd
from isocompy.data_preparation import preprocess
from isocompy.reg_model import model
from isocompy.tools import session, evaluation, stats, plots

#importing data
d=join("case_studies","north_chile","manual_filtered_data")
dir_inp=join(d,"inputs")
dir_outs=join(d,"outputs")

ddir_inp_classess=join(dir_outs,"input_data_classess")

dir_outs_st1=join(dir_outs,"st1_all_month")

dir_outs_st2=join(dir_outs,"st2_1_2_3")

#Temperature data
temp=pd.read_excel(join(dir_inp,"temp_monthly.xls"),sheet_name="temp_monthly",header=0,index_col=False,keep_default_na=True)
#relative humidity data
hum=pd.read_excel(join(dir_inp,"hum_monthly.xls"),sheet_name="hum_monthly",header=0,index_col=False,keep_default_na=True)
#precipitation data
rain=pd.read_excel(join(dir_inp,"rain_monthly.xls"),sheet_name="rain_monthly",header=0,index_col=False,keep_default_na=True)

#isotope data: They are not going to be used until stage 2 models.
data_file_iso = join(dir_inp,"Isotopes.xls")
iso_18 = pd.read_excel(data_file_iso,sheet_name="ISOT18O",header=0,index_col=False,keep_default_na=True)
iso_2h=pd.read_excel(data_file_iso,sheet_name="ISOT2H",header=0,index_col=False,keep_default_na=True)
#iso_3h=pd.read_excel(data_file_iso,sheet_name="ISOT3",header=0,index_col=False,keep_default_na=True)
#-------------------------------------------

#data class

#Precipitation
preped_prc=preprocess()
preped_prc.fit(inp_var=rain,var_name="prc",fields=["CooX","CooY","CooZ"],remove_outliers=False,direc=ddir_inp_classess)

#Temperature
preped_tmp=preprocess()
preped_tmp.fit(inp_var=temp,var_name="tmp",fields=["CooX","CooY","CooZ"],remove_outliers=False,direc=ddir_inp_classess)

#Humidity
preped_hmd=preprocess()
preped_hmd.fit(inp_var=hum,var_name="hmd",fields=["CooX","CooY","CooZ"],remove_outliers=False,direc=ddir_inp_classess)

#isotopes
prep_st2_iso1=preprocess()
prep_st2_iso1.fit(inp_var=iso_18,var_name="iso_18",fields=["CooX","CooY","CooZ"],remove_outliers=False,direc=ddir_inp_classess)

prep_st2_iso2=preprocess()
prep_st2_iso2.fit(inp_var=iso_2h,var_name="iso_2h",fields=["CooX","CooY","CooZ"],remove_outliers=False,direc=ddir_inp_classess)

#prep_st2_iso3=preprocess()
#prep_st2_iso3.fit(inp_var=iso_2h,var_name="is3",fields=["CooX","CooY","CooZ"],remove_outliers=False,direc=ddir_inp_classess)
#-------------------------------------------

#stage 1 model
est_class=model()
est_class.st1_fit(var_cls_list=[preped_prc,preped_tmp,preped_hmd],
                    st1_model_month_list=[1,2,3],
                    direc=dir_outs_st1)
#-------------------------------------------

#stage 1 model plots
plots.best_estimator_plots(est_class,st2=False)
plots.partial_dep_plots(est_class,st2=False)
#-------------------------------------------
#stage 1 prediction
est_class.st1_predict(cls_list=[prep_st2_iso1,prep_st2_iso2],st2_model_month_list=[1,2,3])
#-------------------------------------------

#save stage 1 model object
est_class_dir=session.save(est_class,name='est_class_st1') 
#-------------------------------------------


