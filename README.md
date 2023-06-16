# Predicting-Power-Outage-Cause
This is a project I completed for DSC80 at UCSD. It is, in some ways, a continuation of a previous project on the same [dataset](https://www.sciencedirect.com/science/article/pii/S2352340918307182#bib6). That previous project focused on EDA and hypothesis testing, while this one focuses on predictive modeling. The previous project can be found [here](https://nliu880.github.io/hurricanes-and-outages/).

## Introduction

This project focuses on identifying a prediction and trying to answer it. We will start with framing the problem, creating a baseline model, creating a more finalized model, and then doing a fairness analysis. 

## Framing the Problem

The prediction problem I ask is: Can we predict the cause of a power outage based on other features?

I will use a multiclass classification model to try and solve this prediction problem. The response variable will be `cause category`; this variable describes the general cause (severe weather, intentional attack, etc) of some power outage, which is what I am trying to predict. My evaluation metric will be the F-1 score. I want to balance both precision and recall. Accuracy in this case could also be used, but from my EDA done in my previous project, I discovered there were large outliers that skewed the data. We want to avoid errors that would come from skewed data in our model metric, so we will use the F-1 score. 

All the data I will be using is or will be known by the end of the power outage (ex outage duration). Some of the data is pre-determined by location, or historical trends and the rest will be determined by the end of the outage. 

We will begin this by performing the same data cleaning steps as previous. I have made a few edits that involve adding the time unit to the `outage duration` columns. One is in minutes (the original) and ones is in hours (which I added in for clarity and intuition). I have also dropped the `cause category detail`, `hurricane name`, and `postal code` columns.  In addition, we also fill in the missing values for certain columns by imputing either the mean or the mode of the group when grouped by `cause category`. The final dataframe is seen below. 

| us state   | nerc region   | climate region     |   anomaly level | climate category   | cause category     |   outage duration (min) |   demand loss mw |   customers affected |   population | start time          | restoration time    | total time      |   outage duration (hrs) |
|:-----------|:--------------|:-------------------|----------------:|:-------------------|:-------------------|------------------------:|-----------------:|---------------------:|-------------:|:--------------------|:--------------------|:----------------|------------------------:|
| Minnesota  | MRO           | East North Central |            -0.3 | normal             | severe weather     |                    3060 |              nan |                70000 |  5.34812e+06 | 2011-07-01 17:00:00 | 2011-07-03 20:00:00 | 2 days 03:00:00 |                   51    |
| Minnesota  | MRO           | East North Central |            -0.1 | normal             | intentional attack |                       1 |              nan |                  nan |  5.45712e+06 | 2014-05-11 18:38:00 | 2014-05-11 18:39:00 | 0 days 00:01:00 |                    0.02 |
| Minnesota  | MRO           | East North Central |            -1.5 | cold               | severe weather     |                    3000 |              nan |                70000 |  5.3109e+06  | 2010-10-26 20:00:00 | 2010-10-28 22:00:00 | 2 days 02:00:00 |                   50    |
| Minnesota  | MRO           | East North Central |            -0.1 | normal             | severe weather     |                    2550 |              nan |                68200 |  5.38044e+06 | 2012-06-19 04:30:00 | 2012-06-20 23:00:00 | 1 days 18:30:00 |                   42.5  |
| Minnesota  | MRO           | East North Central |             1.2 | warm               | severe weather     |                    1740 |              250 |               250000 |  5.48959e+06 | 2015-07-18 02:00:00 | 2015-07-19 07:00:00 | 1 days 05:00:00 |                   29    |


## Baseline Model

We will start by splitting the data into training and testing sets. We will then create a basic `DecisionTreeClassifier()` to classify the cause of the outage based on `climate region` and `climate category`. The `climate region` refers to regions designated by National Centers for Environmental Information (Northeast, South, West, etc). The `climate category` column is based on the ONI El Niño/La Niña index that represents how warm/cold a season is, and is the index based off the total year. 

Extreme `climate category` is more likely to correlate with extreme weather and `climate region` can also be an indicator of weather habits that affect outages. Additionally, public officials may deliberately choose to shut off the power grid during certain weather patterns.

The features used in this initial model are `climate category` and `climate region`, both of which are categorical and nominal. Both are encoded using `OneHotEncoder()` from `sklearn`. The decision tree will have a max_depth of 3.

After creating a `Pipeline` object to help transform the data, we fit the model and calculate the F-1 score for both the training and testing data.

- For the training data, the F-1 score is 0.49595961205501965
- For the testing data, the F-1 score is 0.4178350638155524

This model seems to fall on the slightly worse side. F-1 score is a combination of recall and precision, and ranges from 0 to 1 and can be described as the ratio of true positives to the sum of true positives, false positives, and false negatives. The F-1 score for both training and testing data don't make it to the halfway mark of 0.5. Clearly the model can be improved. 

### NMAR Analysis

In this portion, I looked back at the original, full, dataframe to analyze the missingness of the column `time of restoration` to see if it would qualify as NMAR, where the missingness of the value would depend on the value itself. I believe this column to be NMAR as the value itself (if it were not missing) could explain the missingness of the data. For example, several of the observations with `time of restoration` missing have `intentional attack` as the cause of the outage. If the attack failed, there would be no time of restoration for power. Thus I conclude that the missing values in `time of restoration` are NMAR. <br>

I can extend that argument and make the conclusion that the missing values of `outage duration` are then MAR (missing at random). If you examine the dataframe, you notice that only the rows where `time of restoration` (and similarly, `date of restoration`) are missing are the values for `outage duration` also missing. This also makes sense logically, as if there is no outage end time, there is no outage duration time. Thus if `time of restoration` is NMAR, then `outage duration` is MAR.

### Missingness Dependency

In this section, I examine the missingness dependency of the column `customers affected`. 

First, I examine the distribution of the `population` column when `customers affected` is missing or not missing. 

<iframe src = "assets/pop cust dist.html" width=800 height=600 frameBorder=0></iframe>

I set up a permutation test to determine if `customers affected` being missing affects the `population` distribution. I take my null hypothesis to be that the distribution of `population` does not depend on `customers affected` and my alternative hypothesis to be that it is. I use a significance level of 0.05. <br>

After running the permutation test and using the difference in means for my test statistic, I plot my results in the following histogram. 

<iframe src = "assets/pop cust diff.html" width=800 height=600 frameBorder=0></iframe>

I calculated my p-value to be 0.174, which means I fail to reject my null hypothesis and that the distribution of `population` is not dependent on the missingness of `customers affected`, making the `customers affected` column MCAR in reference to the `population` column. <br>

I do a similar setup and test for the relationship between the `demand loss mw` column (peak consumer demand lost in Megawatts) and `customers affected` column using the total variation distance, only this time I find my p-value to be 0. I can then safely reject my null hypothesis for this permutation test (that the distribution of `demand loss mw` was not dependent on the missingness of `customers affected`) in favor of the alternative, that they are dependent. <br>

With the missingness analyzed, I move onto hypothesis testing for my question. Do hurricanes cause longer power outages in comparison to other outages caused by severe weather?

## Hypothesis Testing

I first filter my dataframe to only outages caused by `severe weather`, then drop the rows with missing `outage duration` values. There are only 19 missing `outage duration` values, thus dropping them are unlikely to affect the distribution. I then set up my null and alternative hypothesis. <br>

Null hypothesis: Hurricanes do not cause longer power outages in comparison to other power outages caused by severe weather. <br>
Alternative hypothesis: Hurricanes do cause longer power outages in comparison to other power outages caused by severe weather. <br>

I have a population (outages caused by severe storms) and a sample (outages caused by hurricanes). My hypothesis test will use the mean of the group (the average of the distribution) as my test statistic and use a p-value of 0.05.

<iframe src = "assets/hypothesis.html" width=800 height=600 frameBorder=0></iframe>

The p-value calculated, which is the probability of a statistic as or more extreme as my test statistic occurring in the wild, is 0.00001. This tells me I should reject my null hypothesis in favor of the alternative. I conclude that hurricanes do cause longer power outages in comparison to those caused by severe weather. <br>
