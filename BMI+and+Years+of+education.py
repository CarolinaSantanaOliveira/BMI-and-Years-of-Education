
# coding: utf-8

# # Do more years of education lead to a lower Body Mass Index?

# Carolina Sant’Ana Oliveira (572376), Florian Schmidt (883729), Zeynep Tanca (878568), January 22nd, 2017
# 
# This notebook shows an easy way to apply a simple regression model in python. In particular we analyse the relationship between education measured in years and an important health parameter, namely the body mass index (BMI), using data from the [easySHARE](http://www.share-project.org/data-access-documentation/easyshare.html) panel collected by Tilburg University’s CentER.
# 
# The python code and data used in this notebook can be found at [github](https://github.com/CarolinaSantanaOliveira/BMI-and-Years-of-Education).

# ## Question

# Does education have a positive influence on body mass index? If so, how strong is the effect? In fact, we want to find out whether people who attended educational institutions longer are more likely to live healthier regarding their way of eating, i.e. whether they tend to have a healthier body mass index. As there is already a lot of literature suggesting a positive relationship between education and health status in general, our null hypothesis of independence, therefore, is: Duration of education has no significant effect on the body mass index of people. Our – one sided – alternative hypothesis is: Duration of education has a positive effect on a healthier body mass index of people.

# ## Motivation

# There are plenty of studies on the relationship between education and health. Indeed, almost all of them show that once people are more educated, i.e. the longer they attended school, they care more about their health, their bodies and lives – generally speaking, they live healthier. One of the most important papers on how education and health are linked is the work of [Feinstein et al (2006)](https://www1.oecd.org/edu/innovation-education/37425753.pdf) published by the OECD. They found “international evidence that education is strongly linked to determinants of health” (Feinstein et al, 2006). According to their findings people with more years of schooling are more likely to have better health and healthier behaviour – even though it is not easy to isolate the effect of education from the influence higher incomes linked to better education have on healthier lifestyles. Multiple other studies show resembling findings. [Lleras-Muney (2004)](http://www.econ.ucla.edu/alleras/research/papers/mortalityrevision2.pdf), for example, prove a causal relation between education and mortality, suggesting that one additional year of schooling may reduce the probability of dying within the next ten years by 3.6 percentage points. A study by [Spasojevic (2003)](http://www.emeraldinsight.com/doi/abs/10.1108/S0573-8555%282010%290000290012) shows that an additional year of schooling decreases the risk of bad health (measured by a standardised bad health index) by 18.5 percent. Moreover, a work of [Hermann et al (2011)](http://bmcpublichealth.biomedcentral.com/articles/10.1186/1471-2458-11-169) analyses the relation of education and BMI, suggesting an inverse correlation between higher body mass index and lower education level.
# 
# The latter study is the starting point for the paper at hand. We focus on the BMI as measure for health as it is able to well reflect people’s eating habits and choices for healthy food and because illnesses due to obesity are rising in most of the world’s industrialised countries. To analyse the relation between years of schooling (duration of education) and body mass index we use panel data of the [easySHARE](http://www.share-project.org/data-access-documentation/easyshare.html) set published by Tilburg University’s CentER taking into account some 20 000 observations from participants of Western European countries.
# 

# ## Method

# To answer our question we decided to analyse survey data from people of Western European countries represented in the [easySHARE](http://www.share-project.org/data-access-documentation/easyshare.html) panel (Austria, Belgium, Denmark, France, Germany, Italy, Netherlands, Sweden and Switzerland). We left out Eastern European countries like Estonia, Czech Republic, Hungary etc. assuming that participants with these nationalities cannot be easily compared to Western European people due to the differences in history of political and education system. We focussed on the second wave as here the number of non-response was the smallest. After cleaning the data there was an overall number of 19 050 participants left to analyse the effect of duration of education (“eduyears_mod”) on body mass index measured in absolute numbers (“bmi_mod”) and classified in four categories (“bmi2_mod”). Being aware of the influence income can have on decisions about healthier lifestyle we additionally control for income percentiles of participants categorised from 1 to 10 with 1 representing people being part of the lowest, i.e. poorest, ten percent of the income distribution and 10 representing the richest ten percent of the income distribution. Additionally we introduced group fixed-effects using dummy variables for each country to avoid differences in outcome due to differing nationalities. Time-fixed effects are not necessary for this regression as we are only analysing observations of the same wave, i.e. time period. 

# ## Answer

# Our conclusion is that there is indeed a positive influence of education on a healthier BMI. In other words the BMI decreases in years of education (indicating a negative relationship between both variables). One additional year of education decreases the BMI on average by when not controlling for covariates and 0.14 when controlling for covariates.

# ## Importing libraries and data

# We use the following python packages to run the model.

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl 
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.colors as colors
from statsmodels.formula.api import ols
import statsmodels.api as sm
get_ipython().magic(u'matplotlib inline')


# ## Description of the data

# The data has been extracted from the [easySHARE](http://www.share-project.org/data-access-documentation/easyshare.html) data set. The data consists of 19050 subjects and includes the BMI, years of education, gender, age and income percentile information for each subject, where gender and country variables are dummy variables. Regarding the country dummies, Austria is the reference country. The average body mass index of the sample population is 26.3702, which corresponds to the overweight category. BMI categories will be explained below. The sample population has a maximum BMI observation of 69.92 and a minimum BMI observation of 12.48. Age of the sample varies in a range between 32 and 104 and has a mean of 65.
# 
# BMI is a measure assessing how much a person’s weight diverges from its desired level, which is calculated as the division of the weight by the square root of the height. BMI categories as given by the [World Health Organization](http://www.who.int/en/) are as follows:
# 
#  * Underweight (1): A BMI level below 18.5 is considered as underweight.
# 
#  * Normal (2): A BMI level between 18.5 and 24.9 is considered as normal weight.
# 
#  * Overweight (3): A BMI level between 24.9 and 29.9 is considered as overweight.
# 
#  * Obese (4): A BMI level above 29.9 is considered as obese.

# In[2]:

data1 = pd.read_excel('/Users/Carolina/Desktop/MESTRADO/Applied Economic Analysis/python assignment/dados.xlsx')

#Creating dummies for each country. Austria works as the baseline country.
country_dummies = pd.get_dummies(data1.Country, prefix='Country').iloc[:, 1:]
data = pd.concat([data1, country_dummies], axis=1)
data.head()


# In[3]:

# print the shape of the DataFrame
data.shape


# In[4]:

data1.describe().transpose()


# The graph below allows us to analyze how the observations are distributed within the four BMI categories. 

# In[6]:

# list with BMIs in categories 1,2,3 and 4
cat1 = []
cat2 = []
cat3 = []
cat4 = []
#condition
cat1 = data['Body_Mass_Index'].where(data['BMI_categorized']==1,)
cat2 = data['Body_Mass_Index'].where(data['BMI_categorized']==2,)
cat3 = data['Body_Mass_Index'].where(data['BMI_categorized']==3,)
cat4 = data['Body_Mass_Index'].where(data['BMI_categorized']==4,)
#deleting NAN
cat1 = cat1[~np.isnan(cat1)]
cat2 = cat2[~np.isnan(cat2)]
cat3 = cat3[~np.isnan(cat3)]
cat4 = cat4[~np.isnan(cat4)]

data_to_plot = [cat1,cat2,cat3,cat4]

# Create a figure
fig = plt.figure()

# Create an axes
ax = fig.add_subplot(111)

# Create the boxplot
bp = ax.boxplot(data_to_plot)

fig.suptitle('BMI and correspondent categories in our sample', fontsize=12, fontweight='bold')
ax.set_xticklabels(['Underweight', 'Normal', 'Overweight', 'Obese'], fontsize=11)
ax.set_ylabel('Body Mass Index', fontsize=11)

plt.grid()
plt.show()


# The following graph shows the proportion of observations in each BMI category

# In[7]:

# Counting the number of observations per each BMI category
data['BMI_categorized'].value_counts()
# Data to plot
labels = 'Underweight', 'Normal', 'Overweight', 'Obese'
sizes = [287, 7407,8059 , 3297]
colors = ['gold', 'green', 'coral', 'grey']
explode = (0, 0.1, 0, 0)  # explode 2nd slice
plt.pie(sizes, explode= explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=False, startangle=100)
 
plt.axis('equal')
plt.show()


# <a id='simple_regression'></a>

# ## Relationship between BMI and Years of Education

# To analyze the relation between duration of education we first run a naive regression without any covariates. Later on you will find a more sophisticated regression taking into account covariates like income, gender, age and country.

# ### Naive Regression

# \begin{align}
# BMI_{i} = \alpha + \beta education_{i} + \epsilon_{i}
# \end{align}

# First, we run a naive regression to see the effect of years of education on BMI. The graph below indicates that there is a negative correlation between the BMI level and years of education of subjects. From the table we can see that one additional year of education lowers the BMI by 0.136. As the p-value is 0.00 the effect is highly significant.

# In[8]:

x = data.Years_of_education
y = data.Body_Mass_Index
fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit) 
# fit_fn is now a function which takes in x and returns an estimate for y

plt.figure()
plt.plot(x,y, 'yo', x, fit_fn(x), '--k')
plt.suptitle('Correlation between BMI and Years of education', fontsize=12, fontweight='bold')
plt.xlabel('Years of education', fontsize=11)
plt.ylabel('Body Mass Index', fontsize=11)

plt.grid()
plt.show()


# <a id='sr'></a>

# In[9]:

lm1 = smf.ols(formula='Body_Mass_Index ~ Years_of_education' , data=data).fit()

print(lm1.summary())


# ## Regression Including Covariates

# \begin{align}
# BMI_{i} = \alpha + \beta education_{i} + \gamma gender_{i} + \varphi age_{i} + \phi group fixed effect_{i} + \epsilon_{i}
# \end{align}

# There are many possible covariates which could have an influence on both the independent and dependent variable. To account for possible influences we run a regression including gender, age, country and income as covariates which can be found below. Beforehand, however, we show two examples for these covariates: The first graph presents that the mean of BMI per years of education of females does not follow the same pattern as the males one, indicating that increasing years of education decreases the BMI for women stronger than for men (between 10 and 20 years of education). The second graph shows that for the comparison of Austria to other countries the pattern is different when years of education are higher than 20.  

# In[5]:

#Creating lists with informations for female and male
xfem = data['Years_of_education'].where(data['Female']==1,)
yfem = data['Body_Mass_Index'].where(data['Female']==1,)
xmale = data['Years_of_education'].where(data['Female']==0,)
ymale = data['Body_Mass_Index'].where(data['Female']==0,)

#deleting NAN
xfem = xfem[~np.isnan(xfem)]
yfem = yfem[~np.isnan(yfem)]
xmale = xmale[~np.isnan(xmale)]
ymale = ymale[~np.isnan(ymale)]

fem = pd.concat([xfem, yfem], axis=1)
male = pd.concat([xmale, ymale], axis=1)

# Mean of BMI per year of education
def cust_mean(grp): 
    grp['mean'] = grp['Body_Mass_Index'].mean()      
    return grp
   
fem = fem.groupby(['Years_of_education']).apply(cust_mean)
fem = fem.sort_values(['Years_of_education'], ascending=[True])
fem = fem.drop_duplicates('Years_of_education')


male = male.groupby(['Years_of_education']).apply(cust_mean)
male = male.sort_values(['Years_of_education'], ascending=[True])
male = male.drop_duplicates('Years_of_education')


plt.plot(fem["Years_of_education"], fem["mean"], 'r--', male["Years_of_education"], male["mean"], 'b-')
plt.suptitle('BMI vs Years of education - Female and Male', fontsize=12, fontweight='bold')
plt.xlabel('Years of education', fontsize=11)
plt.ylabel('Body Mass Index', fontsize=11)

plt.annotate('Female', xy=(15,24.8),xytext=(12,24),size=11.5, arrowprops=dict(arrowstyle="->",linewidth=1.4,connectionstyle="arc3,rad=-0.4"),)
plt.annotate('Male', xy=(15,26.34),xytext=(12,27),size=11.5, arrowprops=dict(arrowstyle="->",linewidth=1.4,connectionstyle="arc3,rad=-0.4"),)

plt.show()


# In[6]:

#Creating lists with informations for Austria and all other countries
country = "Austria"
xned = data['Years_of_education'].where(data['Country']==country,)
yned = data['Body_Mass_Index'].where(data['Country']==country,)
xothers = data['Years_of_education'].where(data['Country']!=country,)
yothers = data['Body_Mass_Index'].where(data['Country']!=country,)

#deleting NAN
xned = xned[~np.isnan(xned)]
yned = yned[~np.isnan(yned)]
xothers = xothers[~np.isnan(xothers)]
yothers = yothers[~np.isnan(yothers)]

ned = pd.concat([xned, yned], axis=1)
others = pd.concat([xothers, yothers], axis=1)

# Mean of BMI per year of education
def cust_mean(grp): 
    grp['mean'] = grp['Body_Mass_Index'].mean()      
    return grp
   
ned = ned.groupby(['Years_of_education']).apply(cust_mean)
ned = ned.sort_values(['Years_of_education'], ascending=[True])
ned = ned.drop_duplicates('Years_of_education')


others = others.groupby(['Years_of_education']).apply(cust_mean)
others = others.sort_values(['Years_of_education'], ascending=[True])
others = others.drop_duplicates('Years_of_education')

plt.plot(ned["Years_of_education"], ned["mean"], 'r--', others["Years_of_education"], others["mean"], 'g-' )
plt.suptitle('BMI vs Years of education - Austria and other countries', fontsize=12, fontweight='bold')
plt.xlabel('Years of education', fontsize=11)
plt.ylabel('Body Mass Index', fontsize=11)

plt.annotate('Austria', xy=(21.5,32.8),xytext=(17,30),size=11.5, arrowprops=dict(arrowstyle="->",linewidth=1.4,connectionstyle="arc3,rad=-0.4"),)
plt.annotate('Other countries', xy=(11,26.34),xytext=(10,23.5),size=11.5, arrowprops=dict(arrowstyle="->",linewidth=1.4,connectionstyle="arc3,rad=-0.4"),)

plt.show()


# Finally, the overall regression below shows the results for the relationship between years of education and BMI controlling for age, gender, country and income. As we can see the effect is even stronger as under the naive regression above. Now, one additional year of education lowers the BMI by 0.145. This effect is hihgly significant. The overall regression curve can be seen in the second graph of the partial regression plot.

# <a id='cr'></a>

# In[12]:

lm = ols('Body_Mass_Index ~ Years_of_education + Female + Age + Income_percentiles + Country_Belgium + Country_Denmark +Country_France +Country_Germany +Country_Greece +Country_Italy +Country_Netherlands +Country_Spain +Country_Sweden +Country_Switzerland' , data=data).fit()

print lm.summary()


# In[18]:

fig = plt.figure(figsize=(12,25))
fig = sm.graphics.plot_partregress_grid(lm, fig=fig)
plt.show()


# # Conclusion

# In the light of the results we got in the regressions with and without covariates, we conclude that increasing years of education decreases BMI. Indeed, people who attended educational institutions longer are more likely to be healthier as reflected by the BMI. We do reject our null hypothesis stating that there is no effect of duration of education on the BMI. One additional year of education brings about an average decrease of 0.13 when not controlling for covariates and 0.14 when doing so.

# # References

# * [Börsch-Supan, A., C. Hunkler, S. Gruber, A. Orban, S. Stuck, M. Brandt (2016). easySHARE. Release version: 5.0.0. SHARE-ERIC. Data set DOI: 10.6103/SHARE.easy.500](http://www.share-project.org/data-access-documentation/easyshare.html)
# 
# * [Feinstein, L., R. Sabates, T. Anderson, A. Sorhaindo, C. Hammond (2006). What are the effects of education on health?](https://www1.oecd.org/edu/innovation-education/37425753.pdf)
# 
# * [Hermann, S. et al (2011). The association of education with body mass index and waist circumference in the EPIC-PANACEA study](http://bmcpublichealth.biomedcentral.com/articles/10.1186/1471-2458-11-169)
# * [Lleras-Muney, A. (2004). The Relationship Between Education and Adult Mortality in the United States](http://www.econ.ucla.edu/alleras/research/papers/mortalityrevision2.pdf)
# 
# * [Spasojevic, J. (2003). Effects of Education on Adult Health in Sweden: Results from a Natural Experiment](http://www.emeraldinsight.com/doi/abs/10.1108/S0573-8555%282010%290000290012)
