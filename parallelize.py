# import the libraries and connect to the database
import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kats.consts import TimeSeriesData
import pmdarima as pm
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from kats.detectors.seasonality import FFTDetector
from  multiprocessing import Pool
warnings.filterwarnings("ignore")

main = pd.read_csv("main.csv")
main.date = pd.to_datetime(main.date)
combinations = list()
file_text = open('all_combinations2.txt', "r")

while True: # read till EOF
    file_line = file_text.readline()
    if not file_line:
        print("End Of File")
        break
    res = eval(file_line)
    combinations.append(res)

file_text.close()

def granularize(df):
    """
    Purpose: Takes in the filtered, aggregated data and carry out pre-processing steps including: create a 
            boolean variable to indicate whether there is promotion or not, map the data to a continous date 
            range (there are missing dates in the current subset of data), handling missing values.

    Parameters: The df from create_subset()
    
    Returns: The new processed dataframe

    Note: 
        - Currently missing data is replaced with 0, can be a potential problem?
        - The boolean variable will be treated as the exogenous variable in SARIMAX model

    """
    granular_df = df
    granular_df.reset_index(inplace = True, drop = True)

    # create a column to indicate whether the family product is on promotion OR not
    granular_df['bool_promotion'] = [0 if x == 0 else 1 for x in granular_df.onpromotion]

    # create continuous date for the data
    continuous_date = pd.DataFrame(pd.date_range(start='1/1/2015', end='15/08/2017'), columns=['date'])
    continuous_date = continuous_date.merge(granular_df, how='left', on='date')

    # cleaning up the missing values -- ask question abt this
    continuous_date.sales = continuous_date.sales.fillna(0)
    continuous_date.onpromotion = continuous_date.onpromotion.fillna(0)
    continuous_date.bool_promotion = continuous_date.bool_promotion.fillna(0)

    continuous_date.set_index('date', inplace = True)
    return continuous_date


def create_subset(df, family = None, store_nbr = None, city=None):
    """
    Purpose: Takes in the original data along with the filters to generate a new dataframe of the daily sales aggregated
            on the given level. The new datafram is then passed into granularize() to be pre-processed
            before fitting the sarimax model

    Parameters: family, store_nbr, city. If a level is set to None, the df is not aggregated on that level. 
            If all is None, the daily sales will be the mean across all stores, all cities and all products.
            (Defaulted to None)
    
    Returns: The new processed dataframe

    """
    # filtering with the provided params
    if family is not None:
        df = df[df.family == family]
    if store_nbr is not None:
        df = df[df.store_nbr == store_nbr]
    if city is not None:
        df = df[df.city == city]
    
    # aggregate the daily sales and promotion
    df_new = df.groupby('date').agg({
        'onpromotion':'sum',
        'sales':'mean'}).reset_index() 
    
    return granularize(df_new)

def fit_sarimax(df):
    """
    Purpose: Fit the sarimax model

    Parameters: df—the fully processed dataframe; seasonal–whether this subset of data has seasonality or not
    
    Returns: The model of type pdarima.ARIMA

    Note: 

    """
    
    # creating a temp df to detect the seasonality period
    temp_df = df.reset_index()
    temp_df = temp_df[['date', 'sales']]
    temp_df = temp_df.rename(columns={"date": "time", "sales": "value"})
    ts = TimeSeriesData(temp_df)
    fft_detector = FFTDetector(ts)
    d = fft_detector.detector() 
    if d['seasonality_presence'] == False:
        m = 1
    elif round(min(d['seasonalities'])) > 30: # detect at most monthly pattern, discard the rest as non-seasonal (TIME + data size)
        m = 1
    else:
        m = round(min(d['seasonalities']))
    
    if m == 1:
        seasonal = False
    else:
        seasonal = True
    sxmodel = pm.auto_arima(df[['sales']], exogenous=df[['bool_promotion']],
                            test='adf', m=m,
                            seasonal=seasonal, trace=True,
                            error_action='ignore',  
                            suppress_warnings=True, 
                            stepwise=True, random_state = 0)
    return sxmodel

def parallelize(i):
    with open("progress.txt", "a+") as file_object:
        # if num % 10 == 0: # write to the file after every 50 model fits
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0 :
            file_object.write("\n")
        # Append text at the end of file
        file_object.write("Combination: ")
        file_object.write(str(i))
        temp_df = create_subset(main, family=i[1], city=i[0], store_nbr=i[2])

        try:
            md = fit_sarimax(temp_df)
            if md.pvalues()['bool_promotion'] <= 0.05:
                result = i
            else:
                result = np.nan
            file_object.write("; orders: ")
            file_object.write(str(md.get_params()['order']))
            file_object.write("|")
            file_object.write(str(md.get_params()['seasonal_order']))
            if md.get_params()['with_intercept']:
                file_object.write("|T")
            else:
                file_object.write("|F")
        except Exception as e: 
            print("Error at {}!".format(i))
            print(e)
            pass
    return result

output = []
def collect_result(val):
    output.append(val)

if __name__ == "__main__":
    res = []
    pool = Pool(processes=4)
    for x in combinations:
        pool.apply_async(parallelize, args=(x,), callback=collect_result)
    pool.close()
    pool.join()
    # save the resulting significant combinations to a file
    textfile = open("all_results.txt", "w")
    for element in output:
        textfile.write(str(element) + "\n")
    textfile.close()