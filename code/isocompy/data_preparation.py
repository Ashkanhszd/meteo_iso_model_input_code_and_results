from isocompy.data_prep import data_preparation_func

class preprocess(object):

    def __init__(self,direc,meteo_input_type="daily_remove_outliers",write_outliers_input=True,year_type="all",
    inc_zeros_rain=False,inc_zeros_temp=True,inc_zeros_hum=True,write_integrated_data=True,q1=0.05,q3=0.95,IQR_rain=True,IQR_temp=False,IQR_hum=False,IQR_rat_rain=3,IQR_rat_temp=3,IQR_rat_hum=3):

        self.meteo_input_type=meteo_input_type
        self.year_type=year_type
        self.inc_zeros_rain=inc_zeros_rain
        self.inc_zeros_temp=inc_zeros_temp
        self.inc_zeros_hum=inc_zeros_hum        
        self.IQR_rain=IQR_rain
        self.IQR_temp=IQR_temp
        self.IQR_hum=IQR_hum
        self.write_outliers_input=write_outliers_input
        self.write_integrated_data=write_integrated_data
        self.IQR_rat_rain=IQR_rat_rain
        self.IQR_rat_temp=IQR_rat_temp
        self.IQR_rat_hum=IQR_rat_hum
        self.q1=q1
        self.q3=q3
        self.direc=direc
    
    def fit(self,rain,temp,hum,iso_18,iso_2h,iso_3h,elnino=None,lanina=None):

        if elnino==None: elnino=[]
        if lanina==None: lanina=[]
        self.iso_18=iso_18
        self.iso_2h=iso_2h
        self.iso_3h=iso_3h
        self.month_grouped_iso_18,self.month_grouped_iso_2h,self.month_grouped_iso_3h,self.month_grouped_hum,self.month_grouped_rain,self.month_grouped_temp,self.rain,self.temp,self.hum=data_preparation_func(rain,temp,hum,iso_18,iso_2h,iso_3h,self.direc,self.meteo_input_type,self.q1,self.q3,self.IQR_rain,self.IQR_temp,self.IQR_hum,self.inc_zeros_rain,self.inc_zeros_temp,self.inc_zeros_hum,self.write_outliers_input,self.write_integrated_data,self.IQR_rat_rain,self.IQR_rat_temp,self.IQR_rat_hum,self.year_type,elnino,lanina)
##########################################################################################
      