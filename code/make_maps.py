#Import libraries
from os.path import join
import pandas as pd
from isocompy.data_preparation import preprocess
from isocompy.reg_model import model
from isocompy.tools import session, evaluation, stats, plots

#importing data

#d=join("case_studies","salar_de_atacama","all_data_auto_filtered")
#d=join("case_studies","salar_de_atacama","all_data_no_filter")
#d=join("case_studies","salar_de_atacama","manual_filtered_data_v4")
d=r"C:\Users\Ash kan\Documents\meteo_iso_model_input_code_and_results\case_studies\salar_de_atacama\manual_filtered_data_v4"
dir_inp=join(d,"inputs")
dir_outs=join(d,"outputs")


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
#is_3=pd.read_excel(data_file_iso,sheet_name="ISOT3",header=0,index_col=False,keep_default_na=True)

#-------------------------------------------
pkldir=r"C:\Users\Ash kan\Documents\meteo_iso_model_input_code_and_results\case_studies\salar_de_atacama\manual_filtered_data_v4\outputs\st1_all_month\est_class_st2_18_Jan_2022_20_56_st1True_st2True.pkl"
est_class=session.load(pkldir)
ddir_inp_classess=join(dir_outs,"input_data_classess")

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
#prep_st2_iso3.fit(inp_var=is_3,var_name="is3",fields=["CooX","CooY","CooZ"],remove_outliers=False,direc=ddir_inp_classess)
#-------------------------------------------
est_class.st1_predict(cls_list=[prep_st2_iso1,prep_st2_iso2],st2_model_month_list=[1,2,3])

#to predict in points with observation:
ev_class_obs=evaluation()

#determine the points with needed features (from the class)
pred_inputs=est_class.all_preds[["CooX","CooY","CooZ","month","ID"]].reset_index()
ev_class_obs.predict(est_class,pred_inputs=pred_inputs,direc=join(dir_outs,"preds_obs"))

#to compare 2 isotopes:                     
plots.isotopes_meteoline_plot(ev_class_obs,est_class,iso_18=iso_18,iso_2h=iso_2h,var_list=['iso_18','iso_2h'],obs_data=True,residplot=True)

#-------------------------------------------               
#to predict in points to generate maps: (no observations)

#read points:
data_file = join(dir_inp,"x_y_z.xls")
pred_inputs_map=pd.read_excel(data_file,sheet_name="x_y_z",header=0,index_col=False,keep_default_na=True)
                     
ev_class_map=evaluation()
ev_class_map.predict(est_class,pred_inputs=pred_inputs_map,direc=join(dir_outs,"preds_map_all_points"))

#to compare 2 isotopes:       
plots.isotopes_meteoline_plot(ev_class_map,est_class,var_list=['iso_18','iso_2h'])

#-------------------------------------------               
#test
session.save(ev_class_map)

ev_class_map=session.load(r"C:\Users\Ash kan\Documents\meteo_iso_model_input_code_and_results\case_studies\salar_de_atacama\manual_filtered_data_v4\outputs\preds_map_all_points\isocompy_saved_object_21_Jan_2022_17_27_st1False_st2False.pkl")

#-------------------------------------------               
# To generate the maps

shp_file=join(dir_inp,"cuenca_shp","Cuenca.shp")

opt_title_list=["Precipitation (mm)","Relative Humidity (%)", "Temperature ($^\circ$C)", f'\N{GREEK SMALL LETTER DELTA}\N{SUPERSCRIPT ONE}\N{SUPERSCRIPT EIGHT}O'+'(\u2030 VSMOW)']
feat_list=["prc","hmd","tmp","iso_18"]
observed_class_list=[preped_prc,preped_hmd,preped_tmp,prep_st2_iso1]

# to create .pngs with a shape file
plots.map_generator(ev_class=ev_class_map,feat_list=feat_list,observed_class_list=observed_class_list,shp_file=shp_file,opt_title_list=opt_title_list)


'''#-------------------------------------------               
#to predict in points to generate maps just in salar: (no observations)

#read points:
data_file = join(dir_inp,"x_y_z_just_salar.xls")
pred_inputs_map=pd.read_excel(data_file,sheet_name="x_y_z",header=0,index_col=False,keep_default_na=True)
                     
ev_class_map=evaluation()
ev_class_map.predict(est_class,pred_inputs=pred_inputs_map,direc=join(dir_outs,"preds_map_just_salar_points"))

#to compare 2 isotopes:       
plots.isotopes_meteoline_plot(ev_class_map,est_class,var_list=['iso_18','iso_2h'])

#-------------------------------------------'''