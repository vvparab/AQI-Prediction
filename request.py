# -*- coding: utf-8 -*-
import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'WS10M':2,'WD10M':2,'PS':2,'RH2M':2,'PRECTOTCORR':2,'T2M_MAX':2,'T2M_MIN':2,'T2M':2, 'PM2.5':2,'NOx':2'NH3':2,'SO2':2,'CO':2,'Ozone':2})
print(r.json())