# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('C:/Users/sabh_/Documents/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Major libraries related to Data handling, Vis and statistics
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
from scipy.stats import normaltest, skew
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA, ARMA
from statsmodels.tsa import stattools

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
import ipykernel as ipk
plt.style.use('ggplot')
sns.set_style('whitegrid')
sns.set(font_scale=1.5)
#%config InlineBackend.figure_format = 'retina'

import warnings
warnings.filterwarnings("ignore")

# Pallets used for visualizations
color= "Spectral"
color_plt = ListedColormap(sns.color_palette(color).as_hex())
color_hist = 'teal'
BOLD = '\033[1m'
END = '\033[0m'
stocks = pd.read_csv('C:/Users/sabh_/Documents/kaggle/input/Tadawul_stcks.csv')
stocks_2 = pd.read_csv('C:/Users/sabh_/Documents/kaggle/input/Tadawul_stcks_23_4.csv')
stocks = stocks_2.append(stocks,ignore_index=True)
stocks.rename(columns={'trading_name ': 'trading_name', 'volume_traded ': 'volume_traded','no_trades ':'no_trades'}, inplace=True)
stocks.head()
print(stocks.head())
array = stocks[stocks['sectoer']=='Health Care']['trading_name'].unique()
print(array)
health_care = stocks[stocks['sectoer']=='Health Care']
health_care['date']= pd.to_datetime(health_care['date'])
health_care.sort_values('date', inplace=True)
health_care = health_care.set_index('date')
health_care.head()
print(health_care)
health_care.info()
health_care.isna().sum()
# it's ok that open, high, and low has some missing values, we're not going to us them anyways.
print(health_care.isna().sum())
plt.figure(figsize=(17, 6))
sns.lineplot(x=health_care.index, y="close", hue="trading_name", markers=True, data=health_care)
plt.title('Closing price of Saudi Stocks in the Healthcare Sector')
plt.ylabel('Closing price ($)')
plt.xlabel('Year')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
plt.grid(False)
plt.show()
stocks[stocks['trading_name']=='CHEMICAL'].tail(1)
print(stocks[stocks['trading_name']=='CHEMICAL'].tail(1))