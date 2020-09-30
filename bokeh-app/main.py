#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Defino IDF
A = 2500
B = 5.2
C = 0.82
def IDF(t_d):
    I = A / (B + t_d**C)
    return I 


# In[3]:


df_IDF = pd.DataFrame()
df_IDF['t_d'] = [5*i for i in range(1, 288+1)] 
df_IDF['I'] = df_IDF.apply(lambda x: IDF(x['t_d']), axis=1)

plt.style.use('ggplot')
# In[4]:


# Tormenta de diseño de 24hs de duración y pulsos de 1 hora
df_pulsos_1 = pd.DataFrame()

df_pulsos_1['t_d'] = [60*i for i in range(1, 25)]

df_pulsos_1['I'] = df_pulsos_1.apply(lambda x: IDF(x['t_d']), axis=1)

df_pulsos_1['P_acum'] = df_pulsos_1['t_d'] * df_pulsos_1['I'] / 60

df_pulsos_1['P_i'] = df_pulsos_1['P_acum'].diff()
df_pulsos_1.loc[0, 'P_i'] = df_pulsos_1.loc[0, 'P_acum']


# In[5]:


# Tormenta de diseño de 24hs de duración y pulsos de 2 horas
df_pulsos_2 = pd.DataFrame()

df_pulsos_2['t_d'] = [120*i for i in range(1, 13)]

df_pulsos_2['I'] = df_pulsos_2.apply(lambda x: IDF(x['t_d']), axis=1)

df_pulsos_2['P_acum'] = df_pulsos_2['t_d'] * df_pulsos_2['I'] / 60

df_pulsos_2['P_i'] = df_pulsos_2['P_acum'].diff()
df_pulsos_2.loc[0, 'P_i'] = df_pulsos_2.loc[0, 'P_acum']


# In[6]:


# Tormenta de diseño de 24hs de duración y pulsos de 6 horas
df_pulsos_6 = pd.DataFrame()

df_pulsos_6['t_d'] = [360*i for i in range(1, 5)]

df_pulsos_6['I'] = df_pulsos_6.apply(lambda x: IDF(x['t_d']), axis=1)

df_pulsos_6['P_acum'] = df_pulsos_6['t_d'] * df_pulsos_6['I'] / 60

df_pulsos_6['P_i'] = df_pulsos_6['P_acum'].diff()
df_pulsos_6.loc[0, 'P_i'] = df_pulsos_6.loc[0, 'P_acum']


# In[7]:


def sort_simmetrically(a):
    b = a[1::2] + a[-2::-2]
    return b


# In[8]:


p_i_1 = df_pulsos_1['P_i'].to_list()
p_i_1 = p_i_1[::-1]
p_sorted_1 = sort_simmetrically(p_i_1)


# In[9]:


df_pulsos_1['p_sorted'] = p_sorted_1


# In[10]:


p_i_2 = df_pulsos_2['P_i'].to_list()
p_i_2 = p_i_2[::-1]
p_sorted_2 = sort_simmetrically(p_i_2)


# In[11]:


df_pulsos_2['p_sorted'] = p_sorted_2


# In[12]:


p_i_6 = df_pulsos_6['P_i'].to_list()
p_i_6 = p_i_6[::-1]
p_sorted_6 = sort_simmetrically(p_i_6)


# In[13]:


df_pulsos_6['p_sorted'] = p_sorted_6


# In[14]:


pal_1 = sns.color_palette('Blues_d', len(df_pulsos_1['p_sorted']))
rank_1 = df_pulsos_1['p_sorted'].argsort().argsort()

pal_2 = sns.color_palette('Blues_d', len(df_pulsos_2['p_sorted']))
rank_2 = df_pulsos_2['p_sorted'].argsort().argsort()

pal_3 = sns.color_palette('Blues_d', len(df_pulsos_6['p_sorted']))
rank_3 = df_pulsos_6['p_sorted'].argsort().argsort()


# In[15]:


# Abstracciones SCS
CN = 85
S = (25400/CN) - 254
Ia = 0.2*S


# In[16]:


df_scs_1 = pd.DataFrame()

df_scs_1['t_d'] = df_pulsos_1['t_d']

df_scs_1['P_i_total'] = df_pulsos_1['p_sorted']

df_scs_1['P_t_acum'] = df_scs_1['P_i_total'].cumsum()


# In[17]:


df_scs_2 = pd.DataFrame()

df_scs_2['t_d'] = df_pulsos_2['t_d']

df_scs_2['P_i_total'] = df_pulsos_2['p_sorted']

df_scs_2['P_t_acum'] = df_scs_2['P_i_total'].cumsum()


# In[18]:


df_scs_6 = pd.DataFrame()

df_scs_6['t_d'] = df_pulsos_6['t_d']

df_scs_6['P_i_total'] = df_pulsos_6['p_sorted']

df_scs_6['P_t_acum'] = df_scs_6['P_i_total'].cumsum()


# In[19]:


def apply_scs(df, Ia, S):
    
    # Abstracción Inicial
    df['Ia_acum'] = Ia
    df['Ia_acum'].where(df['P_t_acum'] > Ia, df['P_t_acum'], inplace=True) #Replaces values when condition is false

    df['Ia_i'] = df['Ia_acum'].diff()
    df.loc[0, 'Ia_i'] = df.loc[0, 'Ia_acum']
    
    # Precipitación Efectiva
    df['P_e_acum'] = 0
    df['P_e_acum'].where(df['P_t_acum'] < Ia , (df['P_t_acum'] - Ia)**2 / (df['P_t_acum'] + 0.8*S), inplace=True)

    df['P_i_efec'] = df['P_e_acum'].diff()
    df.loc[0, 'P_i_efec'] = df.loc[0, 'P_e_acum']
    
    # Abstracción Continuada
    df['Fa_acum'] = 0
    df['Fa_acum'].where(df['P_t_acum'] < Ia , (df['P_t_acum'] - Ia)*S / (df['P_t_acum'] + 0.8*S), inplace=True)

    df['Fa_i'] = df['Fa_acum'].diff()
    df.loc[0, 'Fa_i'] = df.loc[0, 'Fa_acum']  


# In[20]:


df_scs_1['Ia_acum'] = Ia
df_scs_1['Ia_acum'].where(df_scs_1['P_t_acum'] > Ia, df_scs_1['P_t_acum'], inplace=True) #Replaces values when condition is false

df_scs_1['Ia_i'] = df_scs_1['Ia_acum'].diff()
df_scs_1.loc[0, 'Ia_i'] = df_scs_1.loc[0, 'Ia_acum']

df_scs_1['P_e_acum'] = 0
df_scs_1['P_e_acum'].where(df_scs_1['P_t_acum'] < Ia , (df_scs_1['P_t_acum'] - Ia)**2 / (df_scs_1['P_t_acum'] + 0.8*S), inplace=True)

df_scs_1['P_i_efec'] = df_scs_1['P_e_acum'].diff()
df_scs_1.loc[0, 'P_i_efec'] = df_scs_1.loc[0, 'P_e_acum']

df_scs_1['Fa_acum'] = 0
df_scs_1['Fa_acum'].where(df_scs_1['P_t_acum'] < Ia , (df_scs_1['P_t_acum'] - Ia)*S / (df_scs_1['P_t_acum'] + 0.8*S), inplace=True)

df_scs_1['Fa_i'] = df_scs_1['Fa_acum'].diff()
df_scs_1.loc[0, 'Fa_i'] = df_scs_1.loc[0, 'Fa_acum']


# In[21]:


df_scs_2['Ia_acum'] = Ia
df_scs_2['Ia_acum'].where(df_scs_2['P_t_acum'] > Ia, df_scs_2['P_t_acum'], inplace=True) #Replaces values when condition is false

df_scs_2['Ia_i'] = df_scs_2['Ia_acum'].diff()
df_scs_2.loc[0, 'Ia_i'] = df_scs_2.loc[0, 'Ia_acum']

df_scs_2['P_e_acum'] = 0
df_scs_2['P_e_acum'].where(df_scs_2['P_t_acum'] < Ia , (df_scs_2['P_t_acum'] - Ia)**2 / (df_scs_2['P_t_acum'] + 0.8*S), inplace=True)

df_scs_2['P_i_efec'] = df_scs_2['P_e_acum'].diff()
df_scs_2.loc[0, 'P_i_efec'] = df_scs_2.loc[0, 'P_e_acum']

df_scs_2['Fa_acum'] = 0
df_scs_2['Fa_acum'].where(df_scs_2['P_t_acum'] < Ia , (df_scs_2['P_t_acum'] - Ia)*S / (df_scs_2['P_t_acum'] + 0.8*S), inplace=True)

df_scs_2['Fa_i'] = df_scs_2['Fa_acum'].diff()
df_scs_2.loc[0, 'Fa_i'] = df_scs_2.loc[0, 'Fa_acum']


# In[22]:


df_scs_6['Ia_acum'] = Ia
df_scs_6['Ia_acum'].where(df_scs_6['P_t_acum'] > Ia, df_scs_6['P_t_acum'], inplace=True) #Replaces values when condition is false

df_scs_6['Ia_i'] = df_scs_6['Ia_acum'].diff()
df_scs_6.loc[0, 'Ia_i'] = df_scs_6.loc[0, 'Ia_acum']

df_scs_6['P_e_acum'] = 0
df_scs_6['P_e_acum'].where(df_scs_6['P_t_acum'] < Ia , (df_scs_6['P_t_acum'] - Ia)**2 / (df_scs_6['P_t_acum'] + 0.8*S), inplace=True)

df_scs_6['P_i_efec'] = df_scs_6['P_e_acum'].diff()
df_scs_6.loc[0, 'P_i_efec'] = df_scs_6.loc[0, 'P_e_acum']

df_scs_6['Fa_acum'] = 0
df_scs_6['Fa_acum'].where(df_scs_6['P_t_acum'] < Ia , (df_scs_6['P_t_acum'] - Ia)*S / (df_scs_6['P_t_acum'] + 0.8*S), inplace=True)

df_scs_6['Fa_i'] = df_scs_6['Fa_acum'].diff()
df_scs_6.loc[0, 'Fa_i'] = df_scs_6.loc[0, 'Fa_acum']


# In[23]:


# Add initial row with null values
columns = ['t_d', 'P_i_total', 'P_t_acum', 'Ia_acum', 'Ia_i', 'P_e_acum', 'P_i_efec', 'Fa_acum', 'Fa_i']
df_null = pd.DataFrame(columns=columns)
df_null.loc[0] = 0

# Append row to dfs
df_scs_1 = df_null.append(df_scs_1, ignore_index=True)
df_scs_2 = df_null.append(df_scs_2, ignore_index=True)
df_scs_6 = df_null.append(df_scs_6, ignore_index=True)


# In[24]:


# Hidrograma unitario instantaneo de Nash

# Características de la cuenca

area = 800 # km2
area_millas2 = (area * 1000**2) / (1609**2) # millas2

L_ppal = 92 # km
L_ppal_millas = (L_ppal*1000) / 1609 # millas

desnivel = 1520 # m
OLS = (desnivel/(L_ppal*1000)) * 10000 # partes por diezmil


# In[25]:


# Estimación de parámetros k y n por Nash-Nash
n_nash = (L_ppal_millas**0.1) / 0.41
k_nash = (27.6/n_nash) * (area_millas2/OLS)**0.3


# In[26]:


from scipy.optimize import minimize #Resuelve calculo iterativo de n


# In[27]:


# Estimar parámetros por Nash-Devoto
k_devoto = 5.9485119 * area**(0.231) * (desnivel/L_ppal)**(-0.777) * (L_ppal**2/area)**(0.124)
tp_devoto = 1.4413144 * area**(0.422) * (desnivel/L_ppal)**(-0.46) * (L_ppal**2/area)**(0.133)

def n_devoto_iter(tp, k):
    
    def objetivo(n_seed):
        n_1 = n_seed
        
        a = 1 + np.sqrt(1/(n_1-1))
        
        b = 1.05 + np.sqrt(1/(n_1-1))

        n_2 = 1 + ((0.05*tp) / (k * (0.05 + np.log(a/b))))
        
        obj = abs(n_2 - n_1)

        return obj
    
    n_seed = 4
    sol = minimize(objetivo, n_seed, method = 'SLSQP', bounds = [(0, 100)])
    return float(sol.x)


# In[28]:


n_devoto = n_devoto_iter(tp_devoto, k_devoto)


# In[29]:


from scipy.special import gamma #Función gamma
from scipy.integrate import quad #Algoritmo de integración


# In[30]:


def hui_nash(t, n, k):
    
    u = (1/k) * (t/k)**(n-1) * np.exp(-t/k) / gamma(n)
    return u 

hui_nash(10, n_devoto, k_devoto)


# In[31]:


df_nash_nash = pd.DataFrame()

df_nash_nash['t'] = [0.1*i for i in range(6000+1)]

df_nash_nash['u'] = hui_nash(df_nash_nash['t'], n_nash, k_nash)


# In[32]:


df_nash_devoto = pd.DataFrame()

df_nash_devoto['t'] = [0.1*i for i in range(6000+1)]

df_nash_devoto['u'] = hui_nash(df_nash_devoto['t'], n_devoto, k_devoto)


# In[33]:


df_nash_nash['u_int_acum'] = np.nan

for j in range (1, len(df_nash_nash)):
    value_nash, error_nash = quad(hui_nash, 0, df_nash_nash.loc[j, 't'], args=(n_nash, k_nash))
    df_nash_nash.loc[j, 'u_int_acum'] = value_nash


# In[34]:


df_nash_devoto['u_int_acum'] = np.nan

for j in range (1, len(df_nash_devoto)):
    value_devoto, error_devoto = quad(hui_nash, 0, df_nash_devoto.loc[j, 't'], args=(n_devoto, k_devoto))
    df_nash_devoto.loc[j, 'u_int_acum'] = value_devoto


# In[35]:


T_1 = 3600

df_HU_1_nash = pd.DataFrame()

df_HU_1_nash['t'] = [i for i in range(600+1)]

df_HU_1_nash['int_c/1'] = np.nan
for j in range(1, len(df_HU_1_nash)):
    value_n1, error_n1 = quad(hui_nash, j-1, j, args=(n_nash, k_nash))
    df_HU_1_nash.loc[j, 'int_c/1'] = value_n1


df_HU_1_nash['int_c/1_/T'] = df_HU_1_nash['int_c/1'] / T_1
df_HU_1_nash['HU_1_m3/s/mm'] = df_HU_1_nash['int_c/1_/T'] * (area*1000**2) / 1000

df_HU_1_devoto = pd.DataFrame()

df_HU_1_devoto['t'] = [i for i in range(600+1)]

df_HU_1_devoto['int_c/1'] = np.nan
for j in range(1, len(df_HU_1_devoto)):
    value_d1, error_d1 = quad(hui_nash, j-1, j, args=(n_devoto, k_devoto))
    df_HU_1_devoto.loc[j, 'int_c/1'] = value_d1


df_HU_1_devoto['int_c/1_/T'] = df_HU_1_devoto['int_c/1'] / T_1
df_HU_1_devoto['HU_1_m3/s/mm'] = df_HU_1_devoto['int_c/1_/T'] * (area*1000**2) / 1000


# In[36]:


T_2 = 3600*2

df_HU_2_nash = pd.DataFrame()

df_HU_2_nash['t'] = [i for i in range(0, 600+1, 2)]

df_HU_2_nash['int_c/2'] = np.nan
for j in range(1, len(df_HU_2_nash)):
    value_n2, error_n2 = quad(hui_nash, j-1, j, args=(n_nash, k_nash))
    df_HU_2_nash.loc[j, 'int_c/2'] = value_n2


df_HU_2_nash['int_c/2_/T'] = df_HU_2_nash['int_c/2'] / T_2
df_HU_2_nash['HU_2_m3/s/mm'] = df_HU_2_nash['int_c/2_/T'] * (area*1000**2) / 1000

df_HU_2_devoto = pd.DataFrame()

df_HU_2_devoto['t'] = [i for i in range(0, 600+1, 2)]

df_HU_2_devoto['int_c/2'] = np.nan
for j in range(1, len(df_HU_2_devoto)):
    value_d2, error_d2 = quad(hui_nash, j-1, j, args=(n_devoto, k_devoto))
    df_HU_2_devoto.loc[j, 'int_c/2'] = value_d2


df_HU_2_devoto['int_c/2_/T'] = df_HU_2_devoto['int_c/2'] / T_2
df_HU_2_devoto['HU_2_m3/s/mm'] = df_HU_2_devoto['int_c/2_/T'] * (area*1000**2) / 1000


# In[37]:


T_6 = 3600*6
df_HU_6_nash = pd.DataFrame()

df_HU_6_nash['t'] = [i for i in range(0, 600+1, 6)]

df_HU_6_nash['int_c/6'] = np.nan
for j in range(1, len(df_HU_6_nash)):
    value_n6, error_n6 = quad(hui_nash, j-1, j, args=(n_nash, k_nash))
    df_HU_6_nash.loc[j, 'int_c/6'] = value_n6


df_HU_6_nash['int_c/6_/T'] = df_HU_6_nash['int_c/6'] / T_6
df_HU_6_nash['HU_6_m3/s/mm'] = df_HU_6_nash['int_c/6_/T'] * (area*1000**2) / 1000

df_HU_6_devoto = pd.DataFrame()

df_HU_6_devoto['t'] = [i for i in range(0, 600+1, 6)]

df_HU_6_devoto['int_c/6'] = np.nan
for j in range(1, len(df_HU_6_devoto)):
    value_d6, error_d6 = quad(hui_nash, j-1, j, args=(n_devoto, k_devoto))
    df_HU_6_devoto.loc[j, 'int_c/6'] = value_d6


df_HU_6_devoto['int_c/6_/T'] = df_HU_6_devoto['int_c/6'] / T_6
df_HU_6_devoto['HU_6_m3/s/mm'] = df_HU_6_devoto['int_c/6_/T'] * (area*1000**2) / 1000


# In[38]:


# Convolución

# Vector U_i
u_1_nash = df_HU_1_nash['HU_1_m3/s/mm'].fillna(0).to_numpy()
u_2_nash = df_HU_2_nash['HU_2_m3/s/mm'].fillna(0).to_numpy()
u_6_nash = df_HU_6_nash['HU_6_m3/s/mm'].fillna(0).to_numpy()

u_1_devoto = df_HU_1_devoto['HU_1_m3/s/mm'].fillna(0).to_numpy()
u_2_devoto = df_HU_2_devoto['HU_2_m3/s/mm'].fillna(0).to_numpy()
u_6_devoto = df_HU_6_devoto['HU_6_m3/s/mm'].fillna(0).to_numpy()


# In[39]:


# Vector pulsos de precipitación efectiva
vector_pulsos_1 = df_scs_1['P_i_efec']
vector_pulsos_2 = df_scs_2['P_i_efec']
vector_pulsos_6 = df_scs_6['P_i_efec']


# In[40]:


# Matriz de pulsos P (sus dimensiones serán distintas para cada vector u_i)
filas_1 = len(vector_pulsos_1)+len(u_1_nash)-1
cols_1 = len(u_1_nash)

filas_2 = len(vector_pulsos_2)+len(u_2_nash)-1
cols_2 = len(u_2_nash)

filas_6 = len(vector_pulsos_6)+len(u_6_nash)-1
cols_6 = len(u_6_nash)


# In[41]:


P_1 = np.zeros((filas_1, cols_1))

for i in range(cols_1):
    
    P_1[i : len(vector_pulsos_1) + i, i] = vector_pulsos_1


P_2 = np.zeros((filas_2, cols_2))

for i in range(cols_2):
    
    P_2[i : len(vector_pulsos_2) + i, i] = vector_pulsos_2


P_6 = np.zeros((filas_6, cols_6))

for i in range(cols_6):
    
    P_6[i : len(vector_pulsos_6) + i, i] = vector_pulsos_6


# In[42]:


Q_d_1_nash = np.dot(P_1, u_1_nash)

Q_d_2_nash = np.dot(P_2, u_2_nash)

Q_d_6_nash = np.dot(P_6, u_6_nash)

Q_d_1_devoto = np.dot(P_1, u_1_devoto)

Q_d_2_devoto = np.dot(P_2, u_2_devoto)

Q_d_6_devoto = np.dot(P_6, u_6_devoto)


# In[43]:


Q_base = 20 #m3/s
df_Qd1_nash = pd.DataFrame({'t' : [i for i in range(len(Q_d_1_nash))],
                       'QD' : Q_d_1_nash,
                       'QT' : Q_d_1_nash + Q_base
                      })

df_Qd2_nash = pd.DataFrame({'t' : [2*i for i in range(len(Q_d_2_nash))],
                       'QD' : Q_d_2_nash,
                       'QT' : Q_d_2_nash + Q_base
                      })

df_Qd6_nash = pd.DataFrame({'t' : [6*i for i in range(len(Q_d_6_nash))],
                       'QD' : Q_d_6_nash,
                       'QT' : Q_d_6_nash + Q_base
                      })

df_Qd1_devoto = pd.DataFrame({'t' : [i for i in range(len(Q_d_1_devoto))],
                       'QD' : Q_d_1_devoto,
                       'QT' : Q_d_1_devoto + Q_base
                      })

df_Qd2_devoto = pd.DataFrame({'t' : [2*i for i in range(len(Q_d_2_devoto))],
                       'QD' : Q_d_2_devoto,
                       'QT' : Q_d_2_devoto + Q_base
                      })

df_Qd6_devoto = pd.DataFrame({'t' : [6*i for i in range(len(Q_d_6_devoto))],
                       'QD' : Q_d_6_devoto,
                       'QT' : Q_d_6_devoto + Q_base
                      })


# In[44]:


# BOKEH (app)
from bokeh.io import output_notebook, show, curdoc
from bokeh.plotting import figure
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, Slider, Select, FactorRange, PreText
from bokeh.transform import dodge, factor_cmap
from bokeh.palettes import Spectral4


# In[45]:


# CDSs

# CDS IDF
source_idf = ColumnDataSource(data={
    't_d' : df_IDF['t_d'],
    'I' : df_IDF['I']
})


# CDS Precipitación Total por Bloques Alternos
source_scs = ColumnDataSource(data={
        't_d' : df_scs_1['t_d'],
        'P_i_total' : df_scs_1['P_i_total'],
        'P_i_efec' : df_scs_1['P_i_efec'] ,
        'Fa_i' : df_scs_1['Fa_i'] ,
        'Ia_i' : df_scs_1['Ia_i'] 
        })

# CDS parámetros HUI
# Define CDS that will change depending on the selected option
x = [('n', 'Nash'), ('n', 'Devoto'), ('k', 'Nash'), ('k', 'Devoto')]
source_nk = ColumnDataSource(data=dict(
    x = x,
    counts = [n_nash, n_devoto, k_nash, k_devoto]
))

# CDS HUI
# Define CDS that will change depending on the selected option
source_hui = ColumnDataSource(data={
    't' : df_nash_nash['t'],
    'u' : df_nash_nash['u']
})

# CDS HU_1, HU_2, HU_6 por Nash y Devoto
# Define CDS that will change depending on the selected option
source_hu = ColumnDataSource(data={
    't' : df_HU_1_nash['t'],
    'int_c/' : df_HU_1_nash['int_c/1'],
    'int_c/_/T' : df_HU_1_nash['int_c/1_/T'],
    'HU_m3/s/mm' : df_HU_1_nash['HU_1_m3/s/mm']   
})

# CDS QD_1, QD_2, QD_6 por Nash y Devoto
# Define CDS that will change depending on the selected option
source_qd = ColumnDataSource(data={
    't' : df_Qd1_nash['t'],
    'QD' : df_Qd1_nash['QD'],
    'QT' : df_Qd1_nash['QT']
})


# In[46]:


# Select
LABELS = ['Pulsos de 1hr', 'Pulsos de 2hr', 'Pulsos de 6hr']

select_pulsos = Select(options=LABELS, value='Pulsos de 1hr', title='Pulsos')
select_hui = Select(options=['Nash', 'Devoto'], value='Nash', title='Parámetros HUI')

# Sliders
slider_CN = Slider(start=1, end=99, value=85, step=1, title='CN', width=500) 
slider_area = Slider(start=100, end=2000, value=area, step=50, title='Área [km^2]') 
slider_lppal = Slider(start=10, end=500, value=L_ppal, step=10, title='Longitud del cauce principal [km]')
slider_desnivel = Slider(start=10, end=3000, value=desnivel, step=10, title='Desnivel [m]')
slider_qbase = Slider(start=10, end=100, value=Q_base, step=10, title='Caudal base [m^3/s]')


# In[47]:


TEXT = PreText(text='')

intro = 'Objetivo: Para una cuenca de caracterísitcas dadas, generar una tormenta de diseño y obtener los pulsos de precipitación efectiva a partir de una función de producción (Método de las abstracciones del SCS).' + '\n' +  'Luego, obtener el hidrograma de caudal directo y total asociados a partir de una función de transferencia (Hidrograma Unitario) y caudal base.' + '\n' + 'Los pasos a seguir son los siguientes:' 

paso_1 = '1. Curva de Intensidad-Duración para un TR dado. Se genera tormenta de 24 hs de duración y pulsos de precipitación total de 1, 2, o 6hs.'

paso_2 = '2. Utilizar el método de los Bloques Alternos para obtener el perfil temporal de la tormenta.'

paso_3 = '3. Obtener los pulsos de precipitación efectiva a partir del método de las abstracciones del SCS.'

paso_4 = '4. Estimación de los parámetros del HUI de Nash a partir del método de Nash o Devoto.'

paso_5 = '5. Obtener la función HUI Nash.'

paso_6 = '6. Obtener el HU_T a partir del HUI siendo T: 1, 2, o 6 hs.'

paso_7 = '7. Convolucionar los pulsos de precipitación efectiva y HU_T. Se obtiene el hidrograma de caudal directo y total (Qtotal = Qdirecto + Qbase).'


TEXT.text = intro + '\n' +  paso_1 + '\n' + paso_2 + '\n' + paso_3 + '\n' + paso_4 + '\n' + paso_5 + '\n' + paso_6 + '\n' + paso_7


# In[48]:


# Plot IDF
p_1 = figure(plot_width=500, plot_height=500, 
             x_axis_label='Duración [min] (eje logarítmico)', y_axis_label='I [mm/h]', title='1. Curva Intensidad Duración',
             x_axis_type='log')

p_1.line(x='t_d', y='I', source=source_idf, color='purple')

p_1.background_fill_color = 'lightblue'
p_1.background_fill_alpha = 0.20


# Plot precipitación total
p_2 = figure(plot_width=500, plot_height=500, 
             x_axis_label='t [minutos]', y_axis_label='P [mm]', title='2. Precipitación Total para: %s' % select_pulsos.value
             )

p_2.vbar(x='t_d', top='P_i_total', source=source_scs, width=20, color='black', fill_color=Spectral4[3], fill_alpha=0.6)

p_2.background_fill_color = 'lightblue'
p_2.background_fill_alpha = 0.20


# Plot SCS
p_3 = figure(plot_width=500, plot_height=500, 
             x_axis_label='t [minutos]', y_axis_label='[mm]', title='3. Abstracciones por SCS para: %s' % select_pulsos.value,
             )

p_3.vbar(x='t_d', top='P_i_total', source=source_scs, color='black', fill_color=Spectral4[3], fill_alpha=0.6, width=20, legend_label='Precipitación Total')
p_3.vbar(x='t_d', top='Ia_i', source=source_scs, color='black', fill_color=Spectral4[1], fill_alpha=0.6, width=20, legend_label='Abstracción Inicial')
p_3.vbar(x='t_d', top='Fa_i', source=source_scs, color='black', fill_color=Spectral4[2], width=20, legend_label='Abstracción Continuada')
p_3.vbar(x='t_d', top='P_i_efec', source=source_scs, color='black', fill_color=Spectral4[0], fill_alpha=0.6, width=20, legend_label='Precipitación Efectiva')

p_3.background_fill_color = 'lightblue'
p_3.background_fill_alpha = 0.20

#Plot n-k
p_4 = figure(plot_width=500, plot_height=500, 
             x_range=FactorRange(*x),
             y_axis_label='[N° de embalses] / [horas]', title='4. Estimación de parámetros'
             )

p_4.vbar(x='x', top='counts',width=0.2, source=source_nk, fill_color='yellow', fill_alpha=0.5, color='black')

p_4.background_fill_color = 'lightblue'
p_4.background_fill_alpha = 0.20

# Plot HUI
p_5 = figure(plot_width=500, plot_height=500,
            x_axis_label='t [hs]', y_axis_label='[1/t]', title='5. Hidrograma Unitario Instantáneo de Nash-%s' % select_hui.value)

p_5.line(x='t', y='u', source=source_hui, color='red')

p_5.background_fill_color = 'lightblue'
p_5.background_fill_alpha = 0.20
p_5.varea(x='t', y1=0, y2='u', source=source_hui, alpha=0.15, fill_color='red')

# Plot HU

p_6 = figure(plot_width=500, plot_height=500,
            x_axis_label='t [hs]', y_axis_label='[m^3/s/mm]', title='6. Hidrograma Unitario asociado a: %s' % select_pulsos.value)

p_6.line(x='t', y='HU_m3/s/mm', source=source_hu, color='purple')

p_6.background_fill_color = 'lightblue'
p_6.background_fill_alpha = 0.20
p_6.varea(x='t', y1=0, y2='HU_m3/s/mm', source=source_hu, alpha=0.15, fill_color='purple')

# Plot QD

p_7 = figure(plot_width=500, plot_height=500,
            x_axis_label='t [hs]', y_axis_label='[m^3/s]', title='7. Hidrogramas de Caudal Directo y Total asociados a: %s' % select_pulsos.value)

p_7.line(x='t', y='QD', source=source_qd, color='brown', legend_label='Caudal Directo')
p_7.line(x='t', y='QT', source=source_qd, color='green', legend_label='Caudal Total')

p_7.background_fill_color = 'lightblue'
p_7.background_fill_alpha = 0.20
p_7.varea(x='t', y1=0, y2='QD', source=source_qd, alpha=0.15, fill_color='brown')
p_7.varea(x='t', y1=0, y2='QT', source=source_qd, alpha=0.15, fill_color='green')


# In[49]:


def callback(attr, old, new):
    
    # Update plot titles
    p_2.title.text = '2. Precipitación Total para: %s' % select_pulsos.value
    p_3.title.text = '3. Abstracciones por SCS para: %s' % select_pulsos.value
    p_5.title.text = '5. Hidrograma Unitario Instantáneo de Nash-%s' % select_hui.value
    p_6.title.text = '6. Hidrograma Unitario asociado a: %s' % select_pulsos.value
    p_7.title.text = '7. Hidrogramas de Caudal Directo y Total asociados a: %s' % select_pulsos.value
    
    
    CN = slider_CN.value
    S = (25400/CN) - 254
    Ia = 0.2*S
    
    # Características de la cuenca
    area = slider_area.value #km2
    area_millas2 = (area * 1000**2) / (1609**2) # millas2
    
    L_ppal = slider_lppal.value #km
    L_ppal_millas = (L_ppal*1000) / 1609 # millas
    
    desnivel = slider_desnivel.value
    OLS = (desnivel/(L_ppal*1000)) * 10000 # partes por diezmil
    
    Q_base = slider_qbase.value
    
    # Estimación de parámetros
    
    # Nash-Nash
    n_nash = (L_ppal_millas**0.1) / 0.41
    k_nash = (27.6/n_nash) * (area_millas2/OLS)**0.3    
    
    # Nash-Devoto
    k_devoto = 5.9485119 * area**(0.231) * (desnivel/L_ppal)**(-0.777) * (L_ppal**2/area)**(0.124)
    tp_devoto = 1.4413144 * area**(0.422) * (desnivel/L_ppal)**(-0.46) * (L_ppal**2/area)**(0.133)
    n_devoto = n_devoto_iter(tp_devoto, k_devoto)
    
    new_parameters = dict(
    x = x,
    counts = [n_nash, n_devoto, k_nash, k_devoto]
    )
    source_nk.data = new_parameters
    
    # Construcción del HUI de Nash

    # HUI Nash-Nash
    df_nash_nash['u'] = hui_nash(df_nash_nash['t'], n_nash, k_nash)

    # HUI Nash-Devoto
    df_nash_devoto['u'] = hui_nash(df_nash_devoto['t'], n_devoto, k_devoto)
    
    if select_hui.value == 'Nash':
        
        new_data_hui = {
        't' : df_nash_nash['t'],
        'u' : df_nash_nash['u'],
    }
        source_hui.data = new_data_hui
        
    if select_hui.value == 'Devoto':
        
        new_data_hui = {
        't' : df_nash_devoto['t'],
        'u' : df_nash_devoto['u'],
    }
        source_hui.data = new_data_hui
    
  
    if select_pulsos.value == 'Pulsos de 1hr':
        
        apply_scs(df_scs_1, Ia, S)
        
        
        new_data = {
        't_d' : df_scs_1['t_d'],
        'P_i_total' : df_scs_1['P_i_total'],
        'P_i_efec' : df_scs_1['P_i_efec'] ,
        'Fa_i' : df_scs_1['Fa_i'] ,
        'Ia_i' : df_scs_1['Ia_i'] 
        }

        source_scs.data = new_data
        
        # Vector pulsos a convolucionar
        vector_pulsos_1 = df_scs_1['P_i_efec']
        
        # Matriz pulsos
        for i in range(cols_1):
    
            P_1[i : len(vector_pulsos_1) + i, i] = vector_pulsos_1
        
        if select_hui.value == 'Nash':
            for j in range(1, len(df_HU_1_nash)):
                value_n1, error_n1 = quad(hui_nash, j-1, j, args=(n_nash, k_nash))
                df_HU_1_nash.loc[j, 'int_c/1'] = value_n1
                df_HU_1_nash['int_c/1_/T'] = df_HU_1_nash['int_c/1'] / T_1
                df_HU_1_nash['HU_1_m3/s/mm'] = df_HU_1_nash['int_c/1_/T'] * (area*1000**2) / 1000

            new_data_hu = {
                't' : df_HU_1_nash['t'],
                'int_c/' : df_HU_1_nash['int_c/1'],
                'int_c/_/T' : df_HU_1_nash['int_c/1_/T'],
                'HU_m3/s/mm' : df_HU_1_nash['HU_1_m3/s/mm']   
            }

            source_hu.data = new_data_hu
            
            # Vector U_i
            u_1_nash = df_HU_1_nash['HU_1_m3/s/mm'].fillna(0).to_numpy()
            # Convolución
            Q_d_1_nash = np.dot(P_1, u_1_nash)
            
            df_Qd1_nash = pd.DataFrame({'t' : [i for i in range(len(Q_d_1_nash))],
                       'QD' : Q_d_1_nash,
                       'QT' : Q_d_1_nash + Q_base
                      })
            new_data_qd = {
                            't' : df_Qd1_nash['t'],
                            'QD' : df_Qd1_nash['QD'],
                            'QT' : df_Qd1_nash['QT']
                            }
            source_qd.data = new_data_qd
       
        if select_hui.value == 'Devoto':
            for j in range(1, len(df_HU_1_devoto)):
                value_d1, error_d1 = quad(hui_nash, j-1, j, args=(n_devoto, k_devoto))
                df_HU_1_devoto.loc[j, 'int_c/1'] = value_d1
                df_HU_1_devoto['int_c/1_/T'] = df_HU_1_devoto['int_c/1'] / T_1
                df_HU_1_devoto['HU_1_m3/s/mm'] = df_HU_1_devoto['int_c/1_/T'] * (area*1000**2) / 1000
            
            new_data_hu = {
                't' : df_HU_1_devoto['t'],
                'int_c/' : df_HU_1_devoto['int_c/1'],
                'int_c/_/T' : df_HU_1_devoto['int_c/1_/T'],
                'HU_m3/s/mm' : df_HU_1_devoto['HU_1_m3/s/mm']   
            }

            source_hu.data = new_data_hu
            
            # Vector U_i
            u_1_devoto = df_HU_1_devoto['HU_1_m3/s/mm'].fillna(0).to_numpy()
            # Convolución
            Q_d_1_devoto = np.dot(P_1, u_1_devoto)
            
            df_Qd1_devoto = pd.DataFrame({'t' : [i for i in range(len(Q_d_1_devoto))],
                       'QD' : Q_d_1_devoto,
                       'QT' : Q_d_1_devoto + Q_base
                      })
            new_data_qd = {
                            't' : df_Qd1_devoto['t'],
                            'QD' : df_Qd1_devoto['QD'],
                            'QT' : df_Qd1_devoto['QT']
                            
                            }
            source_qd.data = new_data_qd
                
                
    
    if select_pulsos.value == 'Pulsos de 2hr':
        
        apply_scs(df_scs_2, Ia, S)
        
        
        new_data = {
        't_d' : df_scs_2['t_d'],
        'P_i_total' : df_scs_2['P_i_total'],
        'P_i_efec' : df_scs_2['P_i_efec'] ,
        'Fa_i' : df_scs_2['Fa_i'] ,
        'Ia_i' : df_scs_2['Ia_i'] 
        }

        source_scs.data = new_data
        
        # Vector pulsos a convolucionar
        vector_pulsos_2 = df_scs_2['P_i_efec']
        
        # Matriz pulsos
        for i in range(cols_2):
    
            P_2[i : len(vector_pulsos_2) + i, i] = vector_pulsos_2
        
        if select_hui.value == 'Nash':
            for j in range(1, len(df_HU_2_nash)):
                value_n2, error_n2 = quad(hui_nash, j-1, j, args=(n_nash, k_nash))
                df_HU_2_nash.loc[j, 'int_c/2'] = value_n2
                df_HU_2_nash['int_c/2_/T'] = df_HU_2_nash['int_c/2'] / T_2
                df_HU_2_nash['HU_2_m3/s/mm'] = df_HU_2_nash['int_c/2_/T'] * (area*1000**2) / 1000


            new_data_hu = {
                            't' : df_HU_2_nash['t'],
                            'int_c/' : df_HU_2_nash['int_c/2'],
                            'int_c/_/T' : df_HU_2_nash['int_c/2_/T'],
                            'HU_m3/s/mm' : df_HU_2_nash['HU_2_m3/s/mm']   
                        }

            source_hu.data = new_data_hu
            
            # Vector U_i
            u_2_nash = df_HU_2_nash['HU_2_m3/s/mm'].fillna(0).to_numpy()
            # Convolución
            Q_d_2_nash = np.dot(P_2, u_2_nash)
            
            df_Qd2_nash = pd.DataFrame({'t' : [2*i for i in range(len(Q_d_2_nash))],
                       'QD' : Q_d_2_nash,
                       'QT' : Q_d_2_nash + Q_base
                      })
            new_data_qd = {
                            't' : df_Qd2_nash['t'],
                            'QD' : df_Qd2_nash['QD'],
                            'QT' : df_Qd2_nash['QT']
                            
                            }
            source_qd.data = new_data_qd
        
        if select_hui.value == 'Devoto':
            
            for j in range(1, len(df_HU_2_devoto)):
                value_d2, error_d2 = quad(hui_nash, j-1, j, args=(n_devoto, k_devoto))
                df_HU_2_devoto.loc[j, 'int_c/2'] = value_d2
                df_HU_2_devoto['int_c/2_/T'] = df_HU_2_devoto['int_c/2'] / T_2
                df_HU_2_devoto['HU_2_m3/s/mm'] = df_HU_2_devoto['int_c/2_/T'] * (area*1000**2) / 1000
            
       
            new_data_hu = {
                            't' : df_HU_2_devoto['t'],
                            'int_c/' : df_HU_2_devoto['int_c/2'],
                            'int_c/_/T' : df_HU_2_devoto['int_c/2_/T'],
                            'HU_m3/s/mm' : df_HU_2_devoto['HU_2_m3/s/mm']   
                        }

            source_hu.data = new_data_hu
            
            # Vector U_i
            u_2_devoto = df_HU_2_devoto['HU_2_m3/s/mm'].fillna(0).to_numpy()
            # Convolución
            Q_d_2_devoto = np.dot(P_2, u_2_devoto)
            
            df_Qd2_devoto = pd.DataFrame({'t' : [2*i for i in range(len(Q_d_2_devoto))],
                       'QD' : Q_d_2_devoto,
                       'QT' : Q_d_2_devoto + Q_base
                      })
            new_data_qd = {
                            't' : df_Qd2_devoto['t'],
                            'QD' : df_Qd2_devoto['QD'],
                            'QT' : df_Qd2_devoto['QT']
                            
                            }
            source_qd.data = new_data_qd

    if select_pulsos.value == 'Pulsos de 6hr':
        
        apply_scs(df_scs_6, Ia, S)
        
    
        new_data = {
        't_d' : df_scs_6['t_d'],
        'P_i_total' : df_scs_6['P_i_total'],
        'P_i_efec' : df_scs_6['P_i_efec'] ,
        'Fa_i' : df_scs_6['Fa_i'] ,
        'Ia_i' : df_scs_6['Ia_i'] 
        }

        source_scs.data = new_data
        
        # Vector pulsos a convolucionar
        vector_pulsos_6 = df_scs_6['P_i_efec']
        
        # Matriz pulsos
        for i in range(cols_6):
    
            P_6[i : len(vector_pulsos_6) + i, i] = vector_pulsos_6
        
        if select_hui.value == 'Nash':
            
            for j in range(1, len(df_HU_6_nash)):
                value_n6, error_n6 = quad(hui_nash, j-1, j, args=(n_nash, k_nash))
                df_HU_6_nash.loc[j, 'int_c/6'] = value_n6
                df_HU_6_nash['int_c/6_/T'] = df_HU_6_nash['int_c/6'] / T_6
                df_HU_6_nash['HU_6_m3/s/mm'] = df_HU_6_nash['int_c/6_/T'] * (area*1000**2) / 1000

            

            new_data_hu = {
                            't' : df_HU_6_nash['t'],
                            'int_c/' : df_HU_6_nash['int_c/6'],
                            'int_c/_/T' : df_HU_6_nash['int_c/6_/T'],
                            'HU_m3/s/mm' : df_HU_6_nash['HU_6_m3/s/mm']   
                        }

            source_hu.data = new_data_hu
            
            # Vector U_i
            u_6_nash = df_HU_6_nash['HU_6_m3/s/mm'].fillna(0).to_numpy()
            # Convolución
            Q_d_6_nash = np.dot(P_6, u_6_nash)
            
            df_Qd6_nash = pd.DataFrame({'t' : [6*i for i in range(len(Q_d_6_nash))],
                       'QD' : Q_d_6_nash,
                       'QT' : Q_d_6_nash + Q_base
                      })
            new_data_qd = {
                            't' : df_Qd6_nash['t'],
                            'QD' : df_Qd6_nash['QD'],
                            'QT' : df_Qd6_nash['QT']
                            }
            source_qd.data = new_data_qd
        
        if select_hui.value == 'Devoto':
            
            for j in range(1, len(df_HU_6_devoto)):
                value_d6, error_d6 = quad(hui_nash, j-1, j, args=(n_devoto, k_devoto))
                df_HU_6_devoto.loc[j, 'int_c/6'] = value_d6
                df_HU_6_devoto['int_c/6_/T'] = df_HU_6_devoto['int_c/6'] / T_6
                df_HU_6_devoto['HU_6_m3/s/mm'] = df_HU_6_devoto['int_c/6_/T'] * (area*1000**2) / 1000
            
            
            
       
            new_data_hu = {
                            't' : df_HU_6_devoto['t'],
                            'int_c/' : df_HU_6_devoto['int_c/6'],
                            'int_c/_/T' : df_HU_6_devoto['int_c/6_/T'],
                            'HU_m3/s/mm' : df_HU_6_devoto['HU_6_m3/s/mm']   
                        }
        
            source_hu.data = new_data_hu
            
            # Vector U_i
            u_6_devoto = df_HU_6_devoto['HU_6_m3/s/mm'].fillna(0).to_numpy()
            # Convolución
            Q_d_6_devoto = np.dot(P_6, u_6_devoto)
            
            df_Qd6_devoto = pd.DataFrame({'t' : [6*i for i in range(len(Q_d_6_devoto))],
                       'QD' : Q_d_6_devoto,
                       'QT' : Q_d_6_devoto + Q_base
                      })
            new_data_qd = {
                            't' : df_Qd6_devoto['t'],
                            'QD' : df_Qd6_devoto['QD'],
                            'QT' : df_Qd6_devoto['QT']
                            
                            }
            source_qd.data = new_data_qd


# In[50]:


# On_changes

# Selects
select_pulsos.on_change('value', callback)
select_hui.on_change('value', callback)

# Sliders
slider_CN.on_change('value', callback)
slider_area.on_change('value', callback)
slider_lppal.on_change('value', callback)
slider_desnivel.on_change('value', callback)
slider_qbase.on_change('value', callback)


# In[51]:


layout_1 = column(select_pulsos, slider_CN,select_hui, slider_area, slider_lppal, slider_desnivel, slider_qbase)
layout = column(TEXT,
row(layout_1, p_1, p_2, p_3),
row(p_4, p_5, p_6, p_7)
      )

curdoc().add_root(layout)

