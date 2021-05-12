#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required packages
import matplotlib as mpl
from matplotlib import colors

get_ipython().run_line_magic('matplotlib', 'inline')
mpl.rcParams['figure.figsize'] = (15, 10) #this increases the inline figure size to 15 tall x 10 wide

import seaborn
import pandas as pd
import geopandas as gpd
import pysal
import numpy as np
import mapclassify
import matplotlib.pyplot as plt
import pylab as pl
import adjustText as aT
import matplotlib as mtp

import warnings
warnings.filterwarnings('ignore') # Change settings so that warnings are not displayed

import contextily as cx
from shapely.geometry import Polygon
from shapely.geometry import Point
import plotly.express as px
from pysal.explore import esda
from pysal.lib import weights
from splot.esda import plot_moran
from splot.esda import moran_scatterplot
from splot.esda import plot_local_autocorrelation
from splot.esda import lisa_cluster
from esda.moran import Moran_Local

# Loading a few new packages
from scipy import stats
from pysal.model import spreg
import statsmodels.formula.api as sm


# In[2]:


hp = gpd.read_file('hp.geojson') #2016 housing points with some additional factors calculated
lb = gpd.read_file('london_boroughs.geojson')
oa = gpd.read_file('OutputAreas.geojson')
sch = gpd.read_file('London_Schools.geojson')


# In[3]:


hp.crs = {'init': 'epsg:27700'}
lb.crs = {'init': 'epsg:27700'}
oa.crs = {'init': 'epsg:27700'}
sch.crs = {'init': 'epsg:27700'}


# In[4]:


oa = oa.rename(columns = {'NAME_2': 'NAME'}, inplace = False)


# In[5]:


hp['Detached'] = np.where(hp['propertytype']=='D', 1, 0)
hp['Flats'] = np.where(hp['propertytype']=='F', 1, 0)
hp['Freehold'] = np.where(hp['duration']=='F', 1, 0)


# In[6]:


hp2 = gpd.sjoin(hp,oa)
pd.set_option('display.max_columns', None)
hp2.head()


# In[7]:


variable_names = ['log_fl_area', 'No_Rooms', 'Flats', 'Detached', 'DEPRHH']


# In[8]:


# Calculating log versions of all distances
hp2['Log_Dist_School'] = np.log(hp2['Dist_School']+1)
hp2['Log_Dist_AcSponsor'] = np.log(hp2['Dist_AcSponsor']+1)
hp2['Log_Dist_FSM'] = np.log(hp2['Dist_FSM']+1)
hp2['Log_Dist_Priv'] = np.log(hp2['Dist_Priv']+1)
hp2['Log_Dist_Of1'] = np.log(hp2['Dist_Of1']+1)
hp2['Log_Dist_Of12'] = np.log(hp2['Dist_Of12']+1)
hp2['Log_Dist_Prim12'] = np.log(hp2['Dist_Prim12']+1)
hp2['Log_Dist_Sec12'] = np.log(hp2['Dist_Sec12']+1)
hp2['Log_Dist_VAS'] = np.log(hp2['Dist_VAS']+1)


# In[9]:


m1 = spreg.OLS(hp2[['logprice']].values, hp2[variable_names].values,
                name_y='logprice', name_x=variable_names, robust='white')
print(m1.summary)


# In[10]:


hp2['residual'] = m1.u
medians = hp2.groupby("NAME").residual.median().to_frame('hood_residual')


# In[11]:


f = plt.figure(figsize=(15,3))
ax = plt.gca()
seaborn.boxplot('NAME', 'residual', ax = ax,
                data=hp2.merge(medians, how='left',
                              left_on='NAME',
                              right_index=True)
                   .sort_values('hood_residual'), palette='bwr')
f.autofmt_xdate()
plt.show()


# In[12]:


w = weights.DistanceBand.from_dataframe(hp2, 500) # Weights based on features within 500 meters


# In[13]:


f = 'logprice ~ ' + ' + '.join(variable_names) + ' + NAME - 1' # Remove intercept (-1) in this fixed effects model
print(f)


# In[14]:


m2 = sm.ols(f, data=hp2).fit()
print(m2.summary2())


# In[15]:


neighborhood_effects = m2.params.filter(like='NAME')
#neighborhood_effects.head()


# In[16]:


stripped = neighborhood_effects.index.str.strip('NAME[').str.strip(']')
neighborhood_effects.index = stripped
neighborhood_effects = neighborhood_effects.to_frame('fixed_effect')
#neighborhood_effects.head()


# In[17]:


neighborhood_effects['NAME'] = neighborhood_effects.index


# In[18]:


base_names = variable_names
dist_names = variable_names + ['Log_Dist_Of1']
# + ['Dist_School']
# + ['Dist_AcSponsor']
# + ['Dist_FSM']
# + ['Dist_Priv']
# + ['Dist_Of1']
# + ['Dist_Of12']
# + ['Dist_Prim12']
# + ['Dist_Sec12']
# + ['Dist_VAS']

# + ['Log_Dist_School']
# + ['Log_Dist_AcSponsor']
# + ['Log_Dist_FSM']
# + ['Log_Dist_Priv']
# + ['Log_Dist_Of1']
# + ['Log_Dist_Of12']
# + ['Log_Dist_Prim12']
# + ['Log_Dist_Sec12']
# + ['Log_Dist_VAS']


# In[19]:


m3 = spreg.OLS(hp2[['logprice']].values, hp2[dist_names].values,
                name_y='logprice', name_x=dist_names, robust='white')
print(m3.summary)


# In[20]:


m4 = spreg.OLS(hp2[['logprice']].values, hp2[dist_names].values,
                name_y='logprice', name_x=dist_names, robust='white', w=w, spat_diag=True, moran=True)
print(m4.summary)


# ## Spatial Lag Model (SAR)

# In[21]:


m5 = spreg.GM_Lag(hp2[['logprice']].values, hp2[dist_names].values,
                     w=w, name_y='logprice', name_x=dist_names, robust='white')
print(m5.summary)


# ## Spatial Error Model (SEM)

# In[22]:


m6 = spreg.GM_Error_Het(hp2[['logprice']].values, hp2[dist_names].values,
                     w=w, name_y='logprice', name_x=dist_names)
print(m6.summary)


# ## Spatial Lag + Error Model (SARMA)

# In[23]:


m7 = spreg.GM_Combo_Het(hp2[['logprice']].values, hp2[dist_names].values,
                     w=w, name_y='logprice', name_x=dist_names)
print(m7.summary)


# ## Spatial Regimes Models

# In[24]:


hpcam = hp2[(hp2["NAME"]=="Camden")]
wcam = weights.DistanceBand.from_dataframe(hpcam, 500)
m7cam = spreg.GM_Error_Het(hpcam[['logprice']].values, hpcam[dist_names].values,
                     w=wcam, name_y='logprice', name_x=dist_names)
print(m7cam.summary)


# In[25]:


hpbd = hp2[(hp2["NAME"]=="Barking and Dagenham")]
wbd = weights.DistanceBand.from_dataframe(hpbd, 500)
m7bd = spreg.GM_Error_Het(hpbd[['logprice']].values, hpbd[dist_names].values,
                     w=wbd, name_y='logprice', name_x=dist_names)
print(m7bd.summary)


# In[26]:


hpken = hp2[(hp2["NAME"]=="Kensington and Chelsea")]
wken = weights.DistanceBand.from_dataframe(hpken, 500)
m7ken = spreg.GM_Error_Het(hpken[['logprice']].values, hpken[dist_names].values,
                     w=wken, name_y='logprice', name_x=dist_names)
print(m7ken.summary)


# In[27]:


hphack = hp2[(hp2["NAME"]=="Hackney")]
whack = weights.DistanceBand.from_dataframe(hphack, 500)
m7hack = spreg.GM_Error_Het(hphack[['logprice']].values, hphack[dist_names].values,
                     w=whack, name_y='logprice', name_x=dist_names)
print(m7hack.summary)


# In[ ]:




