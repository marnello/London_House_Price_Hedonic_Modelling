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


f, ax = plt.subplots(1, figsize=(25, 20)) 

lb.to_crs('EPSG:3857').plot(ax=ax, alpha=0.7, color='None', linewidth=2, edgecolor='grey')
sch.to_crs('EPSG:3857').plot(ax=ax, alpha=0.2, color='navy', markersize=5)
ax.set_axis_off()
#ax.set_title('Total Number of Schools') 
plt.axis('equal') 

cx.add_basemap(ax, source=cx.providers.CartoDB.VoyagerNoLabels, alpha=1)
cx.add_basemap(ax, source=cx.providers.CartoDB.VoyagerOnlyLabels)

#texts = []
#for x, y, label in zip(lb_points.geometry.x, lb_points.geometry.y, lb_points["NAME"]):
#    texts.append(pl.text(x, y, label, fontsize = 16, bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none')))
#    
#aT.adjust_text(texts, force_points=0.7, force_text=0.8, expand_points=(1,1), expand_text=(1,1), 
#               arrowprops=dict(arrowstyle="-", color='grey', lw=0.5))


# In[7]:


bx = seaborn.violinplot(x=hp["Dist_School"])
bx = seaborn.violinplot(x=hp["Dist_AcSponsor"])
bx = seaborn.violinplot(x=hp["Dist_FSM"])
bx = seaborn.violinplot(x=hp["Dist_Of1"])
bx = seaborn.violinplot(x=hp["Dist_Of12"])
bx = seaborn.violinplot(x=hp["Dist_Prim12"])
bx = seaborn.violinplot(x=hp["Dist_Sec12"])
bx = seaborn.violinplot(x=hp["Dist_VAS"])


# In[ ]:





# In[ ]:





# In[8]:


hp2 = gpd.sjoin(hp,oa)
pd.set_option('display.max_columns', None)
hp2.head()


# In[9]:


variable_names = ['log_fl_area', 'No_Rooms', 'Flats', 'Detached', 'DEPRHH']


# In[10]:


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


# In[11]:


bx = seaborn.violinplot(x=hp2["Log_Dist_School"])
bx = seaborn.violinplot(x=hp2["Log_Dist_AcSponsor"])
bx = seaborn.violinplot(x=hp2["Log_Dist_FSM"])
bx = seaborn.violinplot(x=hp2["Log_Dist_Of1"])
bx = seaborn.violinplot(x=hp2["Log_Dist_Of12"])
bx = seaborn.violinplot(x=hp2["Log_Dist_Prim12"])
bx = seaborn.violinplot(x=hp2["Log_Dist_Sec12"])
bx = seaborn.violinplot(x=hp2["Log_Dist_VAS"])


# In[12]:


m1 = spreg.OLS(hp2[['logprice']].values, hp2[variable_names].values,
                name_y='logprice', name_x=variable_names, robust='white')
print(m1.summary)


# In[13]:


hp2['residual'] = m1.u
medians = hp2.groupby("NAME").residual.median().to_frame('hood_residual')


# In[14]:


f = plt.figure(figsize=(15,3))
ax = plt.gca()
seaborn.boxplot('NAME', 'residual', ax = ax,
                data=hp2.merge(medians, how='left',
                              left_on='NAME',
                              right_index=True)
                   .sort_values('hood_residual'), palette='bwr')
f.autofmt_xdate()
plt.show()


# In[15]:


w = weights.DistanceBand.from_dataframe(hp2, 500) # Weights based on features within 500 meters


# In[16]:


lag_residual = weights.spatial_lag.lag_spatial(w, m1.u)
ax = seaborn.regplot(m1.u.flatten(), lag_residual.flatten(), 
                     line_kws=dict(color='orangered'),
                     ci=None)
ax.set_xlabel('Model Residuals - $u$')
ax.set_ylabel('Spatial Lag of Model Residuals - $W u$');


# In[17]:


outliers = esda.moran.Moran_Local(m1.u, w, permutations=999)
error_clusters = (outliers.q % 2 == 1) # only the cluster cores
error_clusters &= (outliers.p_sim <= .001) # filtering out non-significant clusters
f, ax = plt.subplots(1, figsize=(25, 20)) #Subplots allows you to draw multiple plots in one figure
hp2.assign(error_clusters = error_clusters,
          local_I = outliers.Is)\
  .query("error_clusters")\
  .sort_values('local_I')\
  .to_crs('EPSG:3857').plot('local_I', cmap='bwr', marker='.', ax=ax)
ax.set_title("Local Moran's I Hotpsots of Under- and Over-Prediction")
ax.set_axis_off() #Remove axes from plot 
plt.axis('equal') #Set x and y axes to be equal size
cx.add_basemap(ax, source=cx.providers.CartoDB.Voyager)


# In[18]:


f = 'logprice ~ ' + ' + '.join(variable_names) + ' + NAME - 1' # Remove intercept (-1) in this fixed effects model
print(f)


# In[19]:


m2 = sm.ols(f, data=hp2).fit()
print(m2.summary2())


# In[20]:


neighborhood_effects = m2.params.filter(like='NAME')
neighborhood_effects.head()


# In[21]:


stripped = neighborhood_effects.index.str.strip('NAME[').str.strip(']')
neighborhood_effects.index = stripped
neighborhood_effects = neighborhood_effects.to_frame('fixed_effect')
neighborhood_effects.head()


# In[22]:


neighborhood_effects['NAME'] = neighborhood_effects.index


# In[23]:


f, ax = plt.subplots(1, figsize=(25, 20)) #Subplots allows you to draw multiple plots in one figure
lb.to_crs('EPSG:3857').plot(ax=ax, color='k', alpha=0.4)
lb.merge(neighborhood_effects, how='left',
                    left_on='NAME', 
                    right_on='NAME')\
                  .dropna(subset=['fixed_effect'])\
                  .to_crs('EPSG:3857').plot('fixed_effect',
                        ax=ax, alpha=.7)

ax.set_title("London Borough Fixed Effects")
ax.set_axis_off() #Remove axes from plot 
plt.axis('equal') #Set x and y axes to be equal size
cx.add_basemap(ax, source=cx.providers.CartoDB.Voyager)

lb["center"] = lb["geometry"].centroid.to_crs('EPSG:3857')
lb_points = lb.copy()
lb_points.set_geometry("center", inplace = True)

texts = []
for x, y, label in zip(lb_points.geometry.x, lb_points.geometry.y, lb_points["NAME"]):
    texts.append(pl.text(x, y, label, fontsize = 16, bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none')))
    
aT.adjust_text(texts, force_points=0.7, force_text=0.8, expand_points=(1,1), expand_text=(1,1), 
               arrowprops=dict(arrowstyle="-", color='grey', lw=0.5))


# In[24]:


base_names = variable_names
dist_names = variable_names

# + ['Log_Dist_School']
# + ['Log_Dist_AcSponsor']
# + ['Log_Dist_FSM']
# + ['Log_Dist_Priv']
# + ['Log_Dist_Of1']
# + ['Log_Dist_Of12']
# + ['Log_Dist_Prim12']
# + ['Log_Dist_Sec12']
# + ['Log_Dist_VAS']


# In[25]:


m3 = spreg.OLS(hp2[['logprice']].values, hp2[dist_names].values,
                name_y='logprice', name_x=dist_names, robust='white')
print(m3.summary)


# In[26]:


lag_residual = weights.spatial_lag.lag_spatial(w, m3.u)
seaborn.regplot(m3.u.flatten(), lag_residual.flatten(), 
                line_kws=dict(color='orangered'),
                ci=None);


# In[27]:


m4 = spreg.OLS(hp2[['logprice']].values, hp2[dist_names].values,
                name_y='logprice', name_x=dist_names, robust='white', w=w, spat_diag=True, moran=True)
print(m4.summary)


# ## Spatial Lag Model (SAR)

# In[28]:


m5 = spreg.GM_Lag(hp2[['logprice']].values, hp2[dist_names].values,
                     w=w, name_y='logprice', name_x=dist_names, robust='white')
print(m5.summary)


# In[29]:


lag_residual = weights.spatial_lag.lag_spatial(w, m5.u)
seaborn.regplot(m5.u.flatten(), lag_residual.flatten(), 
                line_kws=dict(color='orangered'),
                ci=None);


# ## Spatial Error Model (SEM)

# In[30]:


m6 = spreg.GM_Error_Het(hp2[['logprice']].values, hp2[dist_names].values,
                     w=w, name_y='logprice', name_x=dist_names)
print(m6.summary)


# In[31]:


lag_residual = weights.spatial_lag.lag_spatial(w, m6.u)
seaborn.regplot(m6.u.flatten(), lag_residual.flatten(), 
                line_kws=dict(color='orangered'),
                ci=None);


# ## Spatial Lag + Error Model (SARMA)

# In[32]:


m7 = spreg.GM_Combo_Het(hp2[['logprice']].values, hp2[dist_names].values,
                     w=w, name_y='logprice', name_x=dist_names)
print(m7.summary)


# In[33]:


#lag_residual = weights.spatial_lag.lag_spatial(w, m7.u)
#seaborn.regplot(m7.u.flatten(), lag_residual.flatten(), 
#                line_kws=dict(color='orangered'),
#                ci=None);


# ## Spatial Regimes Models

# In[34]:


hpcam = hp2[(hp2["NAME"]=="Camden")]
wcam = weights.DistanceBand.from_dataframe(hpcam, 500)
m7cam = spreg.GM_Error_Het(hpcam[['logprice']].values, hpcam[dist_names].values,
                     w=wcam, name_y='logprice', name_x=dist_names)
print(m7cam.summary)


# In[35]:


hpbd = hp2[(hp2["NAME"]=="Barking and Dagenham")]
wbd = weights.DistanceBand.from_dataframe(hpbd, 500)
m7bd = spreg.GM_Error_Het(hpbd[['logprice']].values, hpbd[dist_names].values,
                     w=wbd, name_y='logprice', name_x=dist_names)
print(m7bd.summary)


# In[36]:


hpken = hp2[(hp2["NAME"]=="Kensington and Chelsea")]
wken = weights.DistanceBand.from_dataframe(hpken, 500)
m7ken = spreg.GM_Error_Het(hpken[['logprice']].values, hpken[dist_names].values,
                     w=wken, name_y='logprice', name_x=dist_names)
print(m7ken.summary)


# In[37]:


hphack = hp2[(hp2["NAME"]=="Hackney")]
whack = weights.DistanceBand.from_dataframe(hphack, 500)
m7hack = spreg.GM_Error_Het(hphack[['logprice']].values, hphack[dist_names].values,
                     w=whack, name_y='logprice', name_x=dist_names)
print(m7hack.summary)










# # Maps

# ## Log House Price

# In[38]:


hpX = hp2.groupby('NAME').agg({'price':['mean','median'], 'ID':'count'})
hpX.columns = ['Mean_Price','Median_Price', 'House_Count']
hpX = hpX.reset_index()

lb2 = lb.merge(hpX)
lb2['House_Dens'] = lb2['House_Count']/(lb2['HECTARES']*.01)


# In[39]:


f, ax = plt.subplots(1, figsize=(25, 20)) 

lb2.to_crs('EPSG:3857').plot(ax=ax, column='Median_Price', alpha=1,legend=True, cmap='Greens', scheme='Quantiles', k=5, edgecolor='black')
ax.set_axis_off()
ax.set_title('Median House Price') 
plt.axis('equal') 

#cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, alpha=0.6)

texts = []
for x, y, label in zip(lb_points.geometry.x, lb_points.geometry.y, lb_points["NAME"]):
    texts.append(pl.text(x, y, label, fontsize = 16, bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none')))
    
aT.adjust_text(texts, force_points=0.7, force_text=0.8, expand_points=(1,1), expand_text=(1,1), 
               arrowprops=dict(arrowstyle="-", color='grey', lw=0.5))


# In[40]:


# Calculating mean ofsted scores and FSM for publicly funded schools
sch_ofsted = sch[sch.Ofsted_Rating != 0]
sch_ofsted2 = gpd.sjoin(sch_ofsted,lb)
sch_ofsted3 = sch_ofsted2.groupby('NAME').agg({'URN':'count','Ofsted_Rating':['mean','median'],'PerFSM':['mean','median']})

sch_ofsted3.columns = ['School_Count','Mean_Ofsted','Median_Ofsted','Mean_FSM','Median_FSM']
sch_ofsted3 = sch_ofsted3.reset_index()

lb2 = lb.merge(sch_ofsted3)
lb2.crs = {'init': 'epsg:27700'}


# ### Plot 1 - Average Ofsted Score

# In[41]:


cmap_reversed = mtp.cm.get_cmap('Greens_r') # reverses colour ramp as the lower the number, the better for Ofsted ratings

f, ax = plt.subplots(1, figsize=(25, 20)) 

lb2.to_crs('EPSG:3857').plot(ax=ax, column='Mean_Ofsted', alpha=1,legend=True, cmap=cmap_reversed, scheme='Quantiles', k=5, edgecolor='black')
ax.set_axis_off()
ax.set_title('Mean Ofsted Score') 
plt.axis('equal') 

#cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, alpha=0.6)

texts = []
for x, y, label in zip(lb_points.geometry.x, lb_points.geometry.y, lb_points["NAME"]):
    texts.append(pl.text(x, y, label, fontsize = 16, bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none')))
    
aT.adjust_text(texts, force_points=0.7, force_text=0.8, expand_points=(1,1), expand_text=(1,1), 
               arrowprops=dict(arrowstyle="-", color='grey', lw=0.5))


# ### Plot 2 - Percentage of Outstanding Ofsted Schools per Borough

# In[42]:


# Calculating percentage of schools with Ofsted 1 ratings
sch_of1 = sch[sch.Ofsted_Rating == 1]
sch_of1_2 = gpd.sjoin(sch_of1,lb)
sch_of1_3 = sch_of1_2.groupby('NAME').agg({'URN':'count'})

sch_of1_3.columns = ['School_Count']
sch_of1_3 = sch_of1_3.reset_index()

sch_of1_3['Percent_Of1_Sch'] = (sch_of1_3['School_Count']/sch_ofsted3['School_Count'])*100

lb_of1 = lb.merge(sch_of1_3)
lb_of1.crs = {'init': 'epsg:27700'}


# In[43]:


f, ax = plt.subplots(1, figsize=(25, 20)) 

lb_of1.to_crs('EPSG:3857').plot(ax=ax, column='Percent_Of1_Sch', alpha=1,legend=True, cmap='Greens', scheme='Quantiles', k=5, edgecolor='black')
ax.set_axis_off()
ax.set_title('Percentage of Schools with Outstanding Ofsted Rating') 
plt.axis('equal')

#cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, alpha=0.6)

texts = []
for x, y, label in zip(lb_points.geometry.x, lb_points.geometry.y, lb_points["NAME"]):
    texts.append(pl.text(x, y, label, fontsize = 16, bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none')))
    
aT.adjust_text(texts, force_points=0.7, force_text=0.8, expand_points=(1,1), expand_text=(1,1), 
               arrowprops=dict(arrowstyle="-", color='grey', lw=0.5))


# ### Plot 3 - Private Schools

# In[44]:


# Calculating count of Private Schools
sch_priv = sch[sch.TYPE == 'Other Independent School']
sch_priv_2 = gpd.sjoin(sch_priv,lb)
sch_priv_3 = sch_priv_2.groupby('NAME').agg({'URN':'count'})

sch_priv_3.columns = ['Private_School_Count']
sch_priv_3 = sch_priv_3.reset_index()

lb_priv = lb.merge(sch_priv_3)
lb_priv.crs = {'init': 'epsg:27700'}

lb_priv['Priv_Sch_Dens'] = lb_priv['Private_School_Count']/(lb_priv['HECTARES']*.01)


# In[45]:


f, ax = plt.subplots(1, figsize=(25, 20)) 

lb_priv.to_crs('EPSG:3857').plot(ax=ax, column='Priv_Sch_Dens', alpha=1,legend=True, cmap='Greens', scheme='Quantiles', k=5, edgecolor='black')
ax.set_axis_off()
ax.set_title('Density of Private Schools (units per sq. km.)') 
plt.axis('equal')

#cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, alpha=0.6)

#texts = []
#for x, y, label in zip(lb_points.geometry.x, lb_points.geometry.y, lb_points["NAME"]):
#    texts.append(pl.text(x, y, label, fontsize = 16, bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none')))
    
#aT.adjust_text(texts, force_points=0.7, force_text=0.8, expand_points=(1,1), expand_text=(1,1), 
 #              arrowprops=dict(arrowstyle="-", color='grey', lw=0.5))


# ### Plot 4 - Voluntary Aided Schools 

# In[46]:


# Calculating count of Voluntary Aided Schools
sch_VAS = sch[sch.TYPE == 'Voluntary Aided School']
sch_VAS_2 = gpd.sjoin(sch_VAS,lb)
sch_VAS_3 = sch_VAS_2.groupby('NAME').agg({'URN':'count'})

sch_VAS_3.columns = ['VAS_Count']
sch_VAS_3 = sch_VAS_3.reset_index()

lb_VAS = lb.merge(sch_VAS_3)
lb_VAS.crs = {'init': 'epsg:27700'}

lb_VAS['VAS_Sch_Dens'] = lb_VAS['VAS_Count']/(lb_VAS['HECTARES']*.01)


# In[47]:


f, ax = plt.subplots(1, figsize=(25, 20)) 

lb_VAS.to_crs('EPSG:3857').plot(ax=ax, column='VAS_Sch_Dens', alpha=0.9,legend=True, cmap='Greens', scheme='Quantiles', k=5, edgecolor='black')
ax.set_axis_off()
ax.set_title('Density of Voluntary Aided Schools (units per sq. km.)') 
plt.axis('equal')

#cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, alpha=0.6)

#texts = []
#for x, y, label in zip(lb_points.geometry.x, lb_points.geometry.y, lb_points["NAME"]):
#    texts.append(pl.text(x, y, label, fontsize = 16, bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none')))
    
#aT.adjust_text(texts, force_points=0.7, force_text=0.8, expand_points=(1,1), expand_text=(1,1), 
#               arrowprops=dict(arrowstyle="-", color='grey', lw=0.5))


# ### Plot 5 - Sponsored Academy Schools 

# In[48]:


# Calculating count of Academy Sponsor Led Schools
sch_Ac = sch[sch.TYPE == 'Academy Sponsor Led']
sch_Ac_2 = gpd.sjoin(sch_Ac,lb)
sch_Ac_3 = sch_Ac_2.groupby('NAME').agg({'URN':'count'})

sch_Ac_3.columns = ['AcSpon_Count']
sch_Ac_3 = sch_Ac_3.reset_index()

sch_Ac = lb.merge(sch_Ac_3)
sch_Ac.crs = {'init': 'epsg:27700'}

sch_Ac['AcSpon_Sch_Dens'] = sch_Ac['AcSpon_Count']/(sch_Ac['HECTARES']*.01)


# In[55]:


f, ax = plt.subplots(1, figsize=(25, 20)) 

sch_Ac.to_crs('EPSG:3857').plot(ax=ax, column='AcSpon_Sch_Dens', alpha=1,legend=True, cmap='Greens', scheme='Quantiles', k=5, edgecolor='black')
ax.set_axis_off()
ax.set_title('Density of Academy Sponsor Led Schools (units per sq. km.)') 
plt.axis('equal')

#cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, alpha=0.6)

texts = []
for x, y, label in zip(lb_points.geometry.x, lb_points.geometry.y, lb_points["NAME"]):
    texts.append(pl.text(x, y, label, fontsize = 16, bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none')))
    
aT.adjust_text(texts, force_points=0.7, force_text=0.8, expand_points=(1,1), expand_text=(1,1), 
               arrowprops=dict(arrowstyle="-", color='grey', lw=0.5))


# ### Plot 6 - Free School Meals

# In[51]:


f, ax = plt.subplots(1, figsize=(25, 20)) #Subplots allows you to draw multiple plots in one figure


lb2.to_crs('EPSG:3857').plot(ax=ax, column='Mean_FSM', legend=True, cmap='Greens', scheme='FisherJenks', k=5, edgecolor='black')
ax.set_axis_off() #Remove axes from plot 
ax.set_title('Mean Free School Meals') #Plot title text
plt.axis('equal') #Set x and y axes to be equal size
#cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, alpha=0.6)

texts = []
for x, y, label in zip(lb_points.geometry.x, lb_points.geometry.y, lb_points["NAME"]):
    texts.append(pl.text(x, y, label, fontsize = 16, bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none')))
    
aT.adjust_text(texts, force_points=0.7, force_text=0.8, expand_points=(1,1), expand_text=(1,1), 
               arrowprops=dict(arrowstyle="-", color='grey', lw=0.5))

