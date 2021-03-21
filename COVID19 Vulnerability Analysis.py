# Name: Alan Danque
# Date: 20200411
# Course: DSC 550
# Week: 6 Exercise Project Part 1

"""
csse_covid_19_daily_reports\
Date.csv
(will need to create an aggregated version of this dataset at the Stateabbr level)
Fields:
Last_Update
Province_State
Country_Region
Confirmed
Deaths
Recovered

us_city_covid_19_vulnerability.csv
(will need to create an aggregated version of this dataset at the Stateabbr level)
Fields:
	stateabbr
	placename
	aspect_name
	value

Need to get a dataset with state abbreviation with state names

zip_code_database.csv
Fields:
	State		(abbreivation)
	County

EducationByStateUrbanRural.csv
Fields:
	Name
	(Create State Abbreviation)
2014-2018_TOTAL

UnemploymentByState.csv
Fields:
	Name
	(Create State Abbreviation)
	Median Household Income (2018)
	2018

PovertyByState.csv
Fields:
	Name
	Percent Total




csse_covid_19_daily_reports\
	Date.csv

us_city_covid_19_vulnerability.csv
zip_code_database.csv
EducationByStateUrbanRural.csv
UnemploymentByState.csv
PovertyByState.csv

"""

import os
import re
import string
import math
import numpy as np
from pandasql import sqldf

import datetime as dt
import pandas as pd
from os import listdir
from os.path import isfile, join
import os
import sys
import fileinput
from os import walk
from glob import glob
from datetime import datetime
from  dateutil.parser import parse
#import pyspark.sql.functions as F
#from pyspark.sql import Window

from glob import glob

import matplotlib
import matplotlib.pyplot as plt
plot_path = os.path.abspath(__file__)


COVID19_DATA_DIR = 'csse_covid_19_daily_reports'
# target_names = ['ham', 'spam']

def clean_covid19_data(COVID19_DATA_DIR):
    # Clean the CDC file column headers to correct them as they have changed over time
    f = []
    for (dirpath, dirnames, filenames) in walk(COVID19_DATA_DIR):
        # print(filenames)
        f.extend(filenames)
        break
    for fname in f:
        pcsvfile = os.path.join(COVID19_DATA_DIR, fname)
        if ".csv" in pcsvfile:
            # print(pcsvfile)
            # Read in the file
            with open(pcsvfile, 'r') as file:
                filedata = file.read()

            # Standardize the column headers as some of the column header names have changed over time.
            filedata = filedata.replace("Province/State", "Province_State").replace("Country/Region", "Country_Region").replace("Last Update", "Last_Update")

            # Write the file out again
            with open(pcsvfile, 'w') as file:
                file.write(filedata)


def get_covid19_data(COVID19_DATA_DIR):
    # Takes in the folder name to read all contained csvs to load to a list of panda dataframes
    # and then concatenate them before returning the consolidated dataframe
    filenames = glob(COVID19_DATA_DIR+'/*.csv')
    all_dfs = []
    for f in filenames:
        df = pd.read_csv(f, usecols=["Province_State", "Country_Region", "Last_Update", "Confirmed", "Deaths", "Recovered"])
        all_dfs.append(df)
    df_out = pd.concat(all_dfs)
    return df_out

def clean_date(PassDate):
    # print(PassDate)
    dt = parse(str(PassDate).replace("T", " "))
    return dt.strftime('%Y-%m-%d')


def get_all_other_data():
    # Loads the following files.
    # disease-burden-by-risk-factor.csv
    # EducationByStateUrbanRural.csv
    # number-of-deaths-by-risk-factor.csv
    # PovertyByState.csv
    # UnemploymentByState.csv
    # us_city_covid_19_vulnerability.csv

    fname = "disease-burden-by-risk-factor.csv"
    diseasefactors = pd.read_csv(fname, usecols=["Entity","Year","Air pollution (outdoor & indoor) (DALYs)","Child wasting (DALYs)","Child stunting (DALYs)","Secondhand smoke (DALYs)","Unsafe sanitation (DALYs)","Unsafe water source (DALYs)","Low physical activity (DALYs)","High cholesterol (DALYs)","Non-exclusive breastfeeding (DALYs)","Outdoor air pollution (DALYs)","Indoor air pollution (DALYs)","Drug use (DALYs)","Diet low in fruits (DALYs)","Iron deficiency (DALYs)","Zinc deficiency (DALYs)","Diet high in salt (DALYs)","Diet low in vegetables (DALYs)","Vitamin A deficiency (DALYs)","Smoking (DALYs)","High blood pressure (DALYs)","High blood sugar (DALYs)","Obesity (DALYs)"])

    fname = "EducationByStateUrbanRural.csv"
    education = pd.read_csv(fname, usecols=["Name","1970_TOTAL","1980_TOTAL","1990_TOTAL","2000_TOTAL","2014-2018_TOTAL"])

    fname = "number-of-deaths-by-risk-factor.csv"
    deathfactors = pd.read_csv(fname, usecols=["Entity","Code","Year","Unsafe water source (deaths)","Poor sanitation (deaths)","No access to handwashing facility (deaths)","Indoor air pollution (deaths)","Non-exclusive breastfeeding (deaths)","Discontinued breastfeeding (deaths)","Child wasting (deaths)","Child stunting (deaths)","Low birth weight (deaths)","Secondhand smoke (deaths)","Alcohol use (deaths)","Drug use (deaths)","Diet low in fruits (deaths)","Diet low in vegetables (deaths)","Unsafe sex (deaths)","Low physical activity (deaths)","High blood sugar (deaths)","High cholesterol (deaths)","Obesity (deaths)","High blood pressure (deaths)","Smoking (deaths)","Iron deficiency (deaths)","Zinc deficiency (deaths)","Vitamin-A deficiency (deaths)","Low bone mineral density (deaths)","Air pollution (outdoor & indoor) (deaths)","Outdoor air pollution (deaths)","Diet low in fiber (deaths)","Diet high in sodium (deaths)","Diet low in legumes (deaths)","Diet low in calcium (deaths)","Diet high in red meat (deaths)","Diet low in whole grains (deaths)","Diet low in nuts and seeds (deaths)","Diet low in seafood omega-3 fatty acids (deaths)"])

    fname = "PovertyByState.csv"
    poverty = pd.read_csv(fname, usecols=["Name","Percent Total"])

    fname = "UnemploymentByState.csv"
    unemployment = pd.read_csv(fname, usecols=["Name","2010","2011","2012","2013","2014","2015","2016","2017","2018","Median Household Income (2018)"])

    fname = "us_city_covid_19_vulnerability.csv"
    vulnerabilities = pd.read_csv(fname, usecols=["stateabbr","placename","aspect_name","value","prank"])

    fname = "State_Abbr_Code_data.csv"
    state_abbr = pd.read_csv(fname, usecols=["State","Abbrev","Code"])

    fname = "CountryAbbr.csv"
    country_abbr = pd.read_csv(fname, usecols=["Name","Code"])
    return diseasefactors, education, deathfactors, poverty, unemployment, vulnerabilities, state_abbr, country_abbr


def show_deathfactors_line_graphs(us_deathfactors):
    # Dynamically takes the columns from the dataframes to generate a line graph for each.
    # Attempt to show all 22 line graphs on one graph
    # us_deathfactors.plot(x='Year')
    # plt.show()

    # define number of rows and columns for subplots
    # make a list of all dataframe columns
    df_list = []
    rpt_list = []
    cnt = 0
    for column in us_deathfactors:
        if column != 'Year' and column != 'Entity' and column != 'Code':
            print("Generating Graph: "+column)
            rpt_list.append(column)
            df1 = us_deathfactors[['Year', column]]
            df1 = df1.set_index('Year')
            df_list.append(df1)
    # Will come back to figure out how to get these all subplotted together
    nrow = len(df_list)
    ncol = 1
    #plt.rcParams['figure.figsize'] = (20, ncol)
    #fig, axes = plt.subplots(nrows=nrow, ncols=ncol)
    count = 0
    for r in range(nrow):
        print(r)
        for c in range(ncol):
            # df_list[count].plot(ax=axes[r, c])
            df_list[count].plot(kind='line')
            count = count + 1
            plt.savefig('Plots/'+ rpt_list[r])
            plt.show()



def transform_covid19data(coviddates, covid19_data_in):
    df_dlist = []
    #print("Dates to loop through")
    for i, Date_Recorded in enumerate(coviddates):
        #print(coviddates[i])
        covid19data_work = covid19_data_in.loc[covid19_data_new['Fix_Date_Recorded'] <= coviddates[i]]

        # turn following into function
        maxdate = covid19data_work['Date_Recorded'].max()
        covid19data_work['Min_Date_Recorded'] = covid19data_work[covid19data_work['Confirmed'] > 0].groupby('Code')['Date_Recorded'].transform('min')
        covid19data_work['Max_Date_Recorded'] = covid19data_work.groupby('Code')['Date_Recorded'].transform('max')
        covid19data_work['Max_Confirmed'] = covid19data_work.groupby('Code')['Confirmed'].transform('max')
        covid19data_work['Max_Deaths'] = covid19data_work.groupby('Code')['Deaths'].transform('max')
        covid19data_work['Max_Recovered'] = covid19data_work.groupby('Code')['Recovered'].transform('max')
        # Get the numberofdays
        covid19data_work['Min_Date_Recorded'] = pd.to_datetime(covid19data_work['Min_Date_Recorded'])
        covid19data_work['Max_Date_Recorded'] = pd.to_datetime(covid19data_work['Max_Date_Recorded'])
        covid19data_work['NumberOfDays'] = (covid19data_work['Max_Date_Recorded'] - covid19data_work['Min_Date_Recorded']).dt.days

        covid19data_work = covid19data_work.loc[covid19data_work['Date_Recorded'] == maxdate]  # .sort_values('Confirmed',ascending=False)
        covid19data_work = covid19data_work[['Code', 'Fix_Date_Recorded', 'Min_Date_Recorded', 'Max_Date_Recorded', 'NumberOfDays', 'Max_Confirmed', 'Max_Deaths', 'Max_Recovered']].copy()

        nan_value = float("NaN")
        covid19data_work.replace("", nan_value, inplace=True)
        covid19data_work.dropna(subset=["NumberOfDays"], inplace=True)
        covid19data_work['Confirmation_Rate'] = (covid19data_work['Max_Confirmed'] / covid19data_work['NumberOfDays']).replace(np.inf, 0).fillna(0)
        covid19data_work['Deaths_Rate'] = (covid19data_work['Max_Deaths'] / covid19data_work['NumberOfDays']).replace(np.inf, 0).fillna(0)
        covid19data_work['Recovered_Rate'] = (covid19data_work['Max_Recovered'] / covid19data_work['NumberOfDays']).replace(np.inf, 0).fillna(0)
        df_dlist.append(covid19data_work)

    covid19data_out = pd.concat(df_dlist)
    covid19data_out = covid19data_out.drop_duplicates().reset_index(drop=True)
    return covid19data_out




def show_diseasefactor_line_graphs(us_diseasefactors):
    # Dynamically takes the columns from the dataframes to generate a line graph for each.
    # Attempt to show all 22 line graphs on one graph
    #us_diseasefactors.plot(x='Year')
    #plt.show()

    # define number of rows and columns for subplots
    # make a list of all dataframe columns
    df_list = []
    rpt_list = []
    cnt = 0
    for column in us_diseasefactors:
        if column != 'Year' and column != 'Entity':
            print("Generating Graph: "+column)
            rpt_list.append(column)
            df1 = us_diseasefactors[['Year', column]]
            df1 = df1.set_index('Year')
            df_list.append(df1)
    # Will come back to figure out how to get these all subplotted together
    nrow = len(df_list)
    ncol = 1
    #plt.rcParams['figure.figsize'] = (20, 10)
    #fig, axes = plt.subplots(nrows=nrow, ncols=ncol)
    count = 0
    for r in range(nrow):
        print(r)
        df_list[r].plot(kind='line')
        plt.savefig('Plots/'+ rpt_list[r])
        plt.show()
        """
        for c in range(ncol):
            # df_list[count].plot(ax=axes[r, c])
            df_list[count].plot(kind='line')
            count = count + 1
            plt.show()
        """

if __name__ == '__main__':
    clean_covid19_data(COVID19_DATA_DIR)
    covid19data = get_covid19_data(COVID19_DATA_DIR)
    # Clean up the multiple date formats that change through the days.
    covid19data['Date_Recorded'] = covid19data['Last_Update'].apply(clean_date)

    print("******************************************************")
    print("COVID19 DATA - all files from folder:csse_covid_19_daily_reports  have been rewritten to covid19data.csv")
    print(covid19data.head())
    print(covid19data.info())
    print(covid19data.describe())

    diseasefactors, education, deathfactors, poverty, unemployment, vulnerabilities, state_abbr, country_abbr = get_all_other_data()

    print("******************************************************")
    print("Disease Factors - disease-burden-by-risk-factor.csv")
    print(diseasefactors.head())
    print(diseasefactors.info())
    print(diseasefactors.describe())

    print("******************************************************")
    print("Education Data - EducationByStateUrbanRural.csv")
    print(education.head())
    print(education.info())
    print(education.describe())

    print("******************************************************")
    print("Death Factors - number-of-deaths-by-risk-factor.csv")
    print(deathfactors.head())
    print(deathfactors.info())
    print(deathfactors.describe())

    print("******************************************************")
    print("Poverty Data - PovertyByState.csv")
    print(poverty.head())
    print(poverty.info())
    print(poverty.describe())

    print("******************************************************")
    print("Unemployment Data - UnemploymentByState.csv")
    print(unemployment.head())
    print(unemployment.info())
    print(unemployment.describe())

    print("******************************************************")
    print("Vulnerabilities Data - us_city_covid_19_vulnerability.csv")
    print(vulnerabilities.head())
    print(vulnerabilities.info())
    print(vulnerabilities.describe())

    print("******************************************************")
    print("state_abbr Data - State_Abbr_Code_data.csv")
    print(state_abbr.head())
    print(state_abbr.info())
    print(state_abbr.describe())

    print("******************************************************")
    print("country_abbr Data - CountryAbbr.csv")
    print(country_abbr.head())
    print(country_abbr.info())
    print(country_abbr.describe())



    print("***********************covid19data_fix*******************************")
    # Get list of unique Date_Recorded
    covid19data['Fix_Date_Recorded'] = pd.to_datetime(covid19data['Date_Recorded'])
    coviddates = covid19data.Fix_Date_Recorded.unique()
    coviddates = sorted(coviddates)  # coviddates.sort_values('Date_Recorded',ascending=False)
    writedates =  pd.DataFrame(coviddates)
    writedates.to_csv('coviddates.csv')
    # Write to csv
    # covid19data.to_csv('covid19data.csv')

    # Join to US States DataFrame and create unique list of State Names and State Abbreviations.
    covid19data_fix = covid19data.merge(state_abbr, left_on='Province_State', right_on='State',
                                        suffixes=('_left', '_right'))
    covid19data_fix = covid19data_fix.drop('Province_State', 1)
    covid19data_fix = covid19data_fix.drop('Last_Update', 1)
    covid19data_fix.rename(columns={'Country_Region': 'Country'}, inplace=True)

    # Update any empty values to 0
    covid19data_fix.fillna(0)
    covid19data_fix.replace('', np.nan, inplace=True)
    covid19data_fix["Confirmed"] = covid19data_fix["Confirmed"].fillna(0)
    covid19data_fix["Deaths"] = covid19data_fix["Deaths"].fillna(0)
    covid19data_fix["Recovered"] = covid19data_fix["Recovered"].fillna(0)
    # covid19data_fix.to_csv('covid19data_fix.csv')
    covid19_data_new = covid19data_fix.copy()
    covid19_data_new = covid19_data_new.drop('Abbrev', 1)
    covid19_data_new = covid19_data_new.drop('Country', 1)
    covid19data_out = transform_covid19data(coviddates, covid19_data_new)
    covid19data_out.rename(columns={'Fix_Date_Recorded': 'Date_Recorded'}, inplace=True)
    covid19data_out.rename(columns={'Min_Date_Recorded': 'First_Case_Date'}, inplace=True)
    covid19data_out = covid19data_out.drop('Max_Date_Recorded', 1)
    covid19data_out.rename(columns={'Max_Confirmed': 'Confirmed_Cases_Count'}, inplace=True)
    covid19data_out.rename(columns={'Max_Confirmed': 'Confirmed_Cases_Count'}, inplace=True)
    covid19data_out.rename(columns={'Max_Deaths': 'Death_Count'}, inplace=True)
    covid19data_out.rename(columns={'Max_Recovered': 'Recovered_Cases_Count'}, inplace=True)
    covid19data_out.rename(columns={'Code': 'State'}, inplace=True)
    # Write to csv
    covid19data_out.to_csv('covid19data_out.csv')

    # Write CSV per Date
    for i, Date_Recorded in enumerate(coviddates):
        dflists_ByDate = []
        for index, row in state_abbr.iterrows():
            dfwork = covid19data_out.loc[(covid19data_out['Date_Recorded'] == Date_Recorded) & (covid19data_out['State'] == row['Code'])]
            dflists_ByDate.append(dfwork)
        dfworkfilename = 'covid19_as_of_'+str(Date_Recorded)[:10]+'_rpt.csv'
        dflists_ByDate_out = pd.concat(dflists_ByDate)
        dflists_ByDate_out.to_csv(dfworkfilename)

    # Write CSV per State
    for index, row in state_abbr.iterrows():
        dflists_ByState = []
        for i, Date_Recorded in enumerate(coviddates):
            dfwork = covid19data_out.loc[(covid19data_out['Date_Recorded'] == Date_Recorded) & (covid19data_out['State'] == row['Code'])]
            dflists_ByState.append(dfwork)
        dfworkfilename = 'covid19_by_State_' +row['Code'] + '_rpt.csv'
        dflists_ByDate_out = pd.concat(dflists_ByState)
        dflists_ByDate_out.to_csv(dfworkfilename)

    maxdate = covid19data_out['Date_Recorded'].max()
    latestCovid19Datafilename = 'covid19_latestavailable_' + str(maxdate)[:10] + '_rpt.csv'
    latestCovid19Data = covid19data_out.loc[(covid19data_out['Date_Recorded'] == maxdate)]
    latestCovid19Data.to_csv(latestCovid19Datafilename)

    """
    print("dflists")
    print(len(dflists))

    nrow = len(dflists)
    ncol = 1
    # plt.rcParams['figure.figsize'] = (20, ncol)
    # fig, axes = plt.subplots(nrows=nrow, ncols=ncol)
    count = 0
    for r in range(nrow):
        print(r)
        for c in range(ncol):
            # df_list[count].plot(ax=axes[r, c])
            df_list[count].plot(kind='line')
            count = count + 1
            plt.savefig('Plots/' + rpt_list[r])
            plt.show()
    """


    print("***********************vulnerabilities_agg*******************************")
    # vulnerabilities   us_city_covid_19_vulnerability.csv      - CITY LEVEL will need to aggregate to the state level
    vulnerabilities_agg = vulnerabilities.groupby(['stateabbr','aspect_name']).agg({'value':['mean','min','max']})
    vulnerabilities_agg.columns = ['vulnerability_value_mean', 'vulnerability_value_min', 'vulnerability_value_max']
    vulnerabilities_agg = vulnerabilities_agg.reset_index()
    print(vulnerabilities_agg)
    vulnerabilities_agg.rename(columns={'aspect_name': 'vulnerability'}, inplace=True)
    vulnerabilities_agg.rename(columns={'stateabbr': 'State_Abbr'}, inplace=True)
    vulnerabilities_agg = vulnerabilities_agg.merge(state_abbr, left_on='State_Abbr', right_on='Code',
              suffixes=('_left', '_right'))
    vulnerabilities_agg = vulnerabilities_agg.drop('Code', 1)
    vulnerabilities_agg = vulnerabilities_agg.drop('Abbrev', 1)
    vulnerabilities_agg.to_csv('vulnerabilities_agg.csv')

    # PIVOT
    #vulnerabilities_agg_pivot = vulnerabilities_agg.pivot(index='State_Abbr', columns='vulnerability', values='vulnerability_value_mean')
    vulnerabilities_agg_pivot = vulnerabilities_agg.pivot(index='State_Abbr', columns='vulnerability')
    """
    vulnerabilities_agg.columns = ['vulnerability_value_mean', 'vulnerability_value_min', 'vulnerability_value_max']
    vulnerabilities_agg = vulnerabilities_agg.reset_index()
    """
    #vulnerabilities_agg_pivot = vulnerabilities_agg_pivot.rename_axis(None)

    vulnerabilities_agg_pivot.to_csv('vulnerabilities_agg_pivot.csv')
    print("NEXT PIVOT THIS DATA SET ATD ALAN REMINDER")

    #vulnerabilities_agg_pivot_header.columns = vulnerabilities_agg_pivot.iloc[1]
    #vulnerabilities_agg_pivot_header.to_csv('vulnerabilities_agg_pivot_header.csv')


    print("************************Death Factors******************************")
    # deathfactors      number-of-deaths-by-risk-factor.csv     - COUNTRY LEVEL
    us_deathfactors = deathfactors[(deathfactors.Entity == 'United States')]
    us_deathfactors.to_csv('us_deathfactors.csv')
    show_deathfactors_line_graphs(us_deathfactors)


    print("************************Disease Factors******************************")
    # us_diseasefactors     disease-burden-by-risk-factor.csv   - COUNTRY LEVEL
    us_diseasefactors = diseasefactors[(diseasefactors.Entity == 'United States')]
    us_diseasefactors.to_csv('us_diseasefactors.csv')
    show_diseasefactor_line_graphs(us_diseasefactors)





    print("************************education******************************")
    # education         EducationByStateUrbanRural.csv
    education_fix = education.merge(state_abbr, left_on='Name', right_on='State',
              suffixes=('_left', '_right'))
    education_fix = education_fix.drop('Name', 1)
    education_fix = education_fix.drop('Abbrev', 1)
    education_fix.rename(columns={'Code': 'State_Abbr'}, inplace=True)

    education_fix.rename(columns={'1970_TOTAL': 'EDUCATION_1970_TOTAL'}, inplace=True)
    education_fix.rename(columns={'1980_TOTAL': 'EDUCATION_1980_TOTAL'}, inplace=True)
    education_fix.rename(columns={'1990_TOTAL': 'EDUCATION_1990_TOTAL'}, inplace=True)
    education_fix.rename(columns={'2000_TOTAL': 'EDUCATION_2000_TOTAL'}, inplace=True)
    education_fix.rename(columns={'2014-2018_TOTAL': 'EDUCATION_2014-2018_TOTAL'}, inplace=True)

    #education_fix.rename(columns={'Country_Region': 'Country'}, inplace=True)
    education_fix.to_csv('education_fix.csv')

    print("************************poverty******************************")
    # poverty           PovertyByState.csv
    poverty_fix = poverty.merge(state_abbr, left_on='Name', right_on='State',
              suffixes=('_left', '_right'))
    poverty_fix = poverty_fix.drop('Name', 1)
    poverty_fix = poverty_fix.drop('Abbrev', 1)
    poverty_fix.rename(columns={'Code': 'State_Abbr'}, inplace=True)
    poverty_fix.rename(columns={'Percent Total': 'POVERTY_PERCENT_TOTAL'}, inplace=True)

    #education_fix.rename(columns={'Country_Region': 'Country'}, inplace=True)
    poverty_fix.to_csv('poverty_fix.csv')

    print("************************unemployment******************************")
    # unemployment      UnemploymentByState.csv
    unemployment_fix = unemployment.merge(state_abbr, left_on='Name', right_on='State',
              suffixes=('_left', '_right'))
    unemployment_fix = unemployment_fix.drop('Name', 1)
    unemployment_fix = unemployment_fix.drop('Abbrev', 1)
    unemployment_fix.rename(columns={'Code': 'State_Abbr'}, inplace=True)
    unemployment_fix.rename(columns={'2010': 'UNEMPLOYMENT_2010'}, inplace=True)
    unemployment_fix.rename(columns={'2011': 'UNEMPLOYMENT_2011'}, inplace=True)
    unemployment_fix.rename(columns={'2012': 'UNEMPLOYMENT_2012'}, inplace=True)
    unemployment_fix.rename(columns={'2013': 'UNEMPLOYMENT_2013'}, inplace=True)
    unemployment_fix.rename(columns={'2014': 'UNEMPLOYMENT_2014'}, inplace=True)
    unemployment_fix.rename(columns={'2015': 'UNEMPLOYMENT_2015'}, inplace=True)
    unemployment_fix.rename(columns={'2016': 'UNEMPLOYMENT_2016'}, inplace=True)
    unemployment_fix.rename(columns={'2017': 'UNEMPLOYMENT_2017'}, inplace=True)
    unemployment_fix.rename(columns={'2018': 'UNEMPLOYMENT_2018'}, inplace=True)
    unemployment_fix.rename(columns={'Median Household Income (2018)': 'UNEMPLOYMENT_Median Household Income (2018)'}, inplace=True)
    unemployment_fix.to_csv('unemployment_fix.csv')


    print("************************MERGE******************************")

    unify1 = unemployment_fix.merge(poverty_fix, left_on='State_Abbr', right_on='State_Abbr',
              suffixes=('_left', '_right'))
    unify2 = unify1.merge(education_fix, left_on='State_Abbr', right_on='State_Abbr',
                                    suffixes=('_left', '_right'))

    unify2 = unify2.drop('State_left', 1)
    unify2 = unify2.drop('State_right', 1)
    unify2.to_csv('unify2.csv')



"""
    plt.rcParams['figure.figsize'] = (20, 10)
    # make subplots
    fig, axes = plt.subplots(nrows=11, ncols=2)

    # make the data read to feed into the visulizer
    X_Axis = us_diseasefactors.replace({'Survived': {1: 'yes', 0: 'no'}}).groupby('Survived').size().reset_index(name='Counts')[
        'Survived']
    Y_Axis = us_diseasefactors.replace({'Survived': {1: 'yes', 0: 'no'}}).groupby('Survived').size().reset_index(name='Counts')[
        'Counts']

    # make the bar plot
    axes[0, 0].bar(X_Axis, Y_Axis)
    axes[0, 0].set_title('Survived', fontsize=25)
    axes[0, 0].set_ylabel('Counts', fontsize=20)
    axes[0, 0].tick_params(axis='both', labelsize=15)

"""




"""
    fig, axes = plt.subplots(5, 1, figsize=(12, 8))

    for i, (j, col) in enumerate(us_diseasefactors.iteritems()):
        ax = axes[i]
        col = col.rename_axis([None, None])
        col.unstack(fill_value=0).plot(ax=ax, title=j, legend=False)

        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)

    fig.tight_layout()

"""

"""
    ax = plt.gca()
    us_diseasefactors.plot(kind='line', x='Year', y='Air pollution (outdoor & indoor) (DALYs)', color='red', ax=ax)
    #us_diseasefactors.plot(kind='line', x='Year', y='Child wasting (DALYs)', color='blue', ax=ax)
    plt.show()
"""

#    us_diseasefactors.to_csv('us_diseasefactors.csv')



"""

Country_Region	Confirmed	Deaths	Recovered	Date_Recorded	State	Abbrev	Code

"""

"""
NEXT get the following loaded / cleaned .

us_city_covid_19_vulnerability.csv
zip_code_database.csv
EducationByStateUrbanRural.csv
UnemploymentByState.csv
PovertyByState.csv
disease-burden-by-risk-factor.csv
number-of-deaths-by-risk-factor.csv



then

review distribution using titanic means
get statistics
evaluate features
evaulate test model
review hyper parameters


"""