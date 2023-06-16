# Predicting-Power-Outage-Cause
created by: Nicole Liu

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

We will start by splitting the data into training and testing sets. We will then create a basic `DecisionTreeClassifier()`, which makes recursive splitting decisions to classify something to a final node, to classify the cause of the outage based on `climate region` and `climate category`. The `climate region` refers to regions designated by National Centers for Environmental Information (Northeast, South, West, etc). The `climate category` column is based on the ONI El Niño/La Niña index that represents how warm/cold a season is, and is the index based off the total year. 

Extreme `climate category` is more likely to correlate with extreme weather and `climate region` can also be an indicator of weather habits that affect outages. Additionally, public officials may deliberately choose to shut off the power grid during certain weather patterns.

The features used in this initial model are `climate category` and `climate region`, both of which are categorical and nominal. Both are encoded using `OneHotEncoder()` from `sklearn`. The decision tree will have a max_depth of 3.

After creating an `sklearn` `Pipeline` object to help transform the data, we fit the model and calculate the F-1 score for both the training and testing data.

- For the training data, the F-1 score is 0.49595961205501965
- For the testing data, the F-1 score is 0.4178350638155524

This model seems to fall on the slightly worse side. F-1 score is a combination of recall and precision, and ranges from 0 to 1 and can be described as the ratio of true positives to the sum of true positives, false positives, and false negatives. The F-1 score for both training and testing data don't make it to the halfway mark of 0.5. Clearly the model can be improved. 

## Final Model

Now, we want to create a better model. In addition to the two features looked at previously, we will also look at `us state`, `anomaly level`, and `outage duration (min)`. We will perform a quantile encoding for `outage duration (min)` and map `us state` to a numerical value using a custom function.

- `us state` contains information about what state an outage occurred in. One potential cause for power outages is severe weather. We may associate certain states with severe weather. Texas, for example, has within recent memory made news headlines both extreme cold and extreme heat. 
- `anomaly level` contains information about the ONI index, a [3 month average temperature difference from the average temperature](https://www.climate.gov/news-features/understanding-climate/climate-variability-oceanic-ni\%C3\%B1o-index). This value is indicative of global temperature changes and can be correlated with weather, which in turn can cause power outages.
- `outage duration (min)` describes how long the power outage was, in minutes. I had found in my previous project that power outages caused by severe weather, hurricanes in particular, were correlated with longer than average power outages. This fact could affect our classification. 

We will use `DecisionTreeClassifier()` again to make our predictions. Additionally, we will also search for the best hyperparameters to use in this model. We will tune the maximum depth of the tree, the minimum number of samples needed to split an internal node, and the criterion for splitting (the function by which the quality of the split is judged). 

- Too deep of a tree may result in overfitting; not deep enough does not allow for detailed enough classification
- The larger the minimum number of samples needed to split an internal node, the more likely the classifier will generalize
- Different criteria for splitting will lead to different decisions and classifications; the ones chosen are the accepted criteria as given in the documentation 

We will find the best hyperparameters by using `GridSearchCV'. Again, we chain together all column transformations and the `DecisionTreeClassifier()` object using an `sklearn` `Pipeline`. We combine these within `GridSearchCV' and use k-fold cross validation, with k = 8, to find the best hyperparameters. We find the following:

- The best criterion is entropy
- The best maximum tree depth is 5
- The best minimum number of samples needed to split a node is 2

After finding these parameters to use in our final model, we fit our final model on the same training and testing data as we used in our baseline model. We find the F-1 scores to be:

- For the training data, the F-1 score is 0.7025791136434512
- For the testing data, the F-1 score is 0.6015811365023541

While this is still not perfect, this is a much better improvement on our previous baseline model. The inclusion of `us state`, `anomaly level`, and `outage duration (min)` has helped us classify power outages to their causes much better.  

## Fairness Analysis

Now we see if the model performs better or worse for certain groups. We will split our data by state into landlocked states and not landlocked states. A quick search (or if a map if you are exceptionally good at geography) tells us that the not landlocked states are: Alaska, Hawai'i, Washington, Oregon, California, Texas, Louisiana, Alabama, Florida, Georgia, South Carolina, North Carolina, Virginia, Maryland, Delaware, New Jersey, Mississippi, New York, Connecticut, Rhode Island, Massachusetts, New Hampshire, and Maine. A few notes on this: 

- Rhode Island has somehow avoided any power outages (or its power outages avoided being included in this set), so we will not include it
- Hawai'i is written as Hawaii in this dataset; we will use Hawaii
- We will add a column that describes whether or not the region in which some power outage occurred was landlocked. A 1 will mean yes, the region is landlocked, and a 0 will represent the opposite

| us state   | nerc region   | climate region     |   anomaly level | climate category   | cause category     |   outage duration (min) |   demand loss mw |   customers affected |   population | start time          | restoration time    | total time      |   outage duration (hrs) |   landlocked |
|:-----------|:--------------|:-------------------|----------------:|:-------------------|:-------------------|------------------------:|-----------------:|---------------------:|-------------:|:--------------------|:--------------------|:----------------|------------------------:|-------------:|
| Minnesota  | MRO           | East North Central |            -0.3 | normal             | severe weather     |                    3060 |              nan |                70000 |  5.34812e+06 | 2011-07-01 17:00:00 | 2011-07-03 20:00:00 | 2 days 03:00:00 |                   51    |            1 |
| Minnesota  | MRO           | East North Central |            -0.1 | normal             | intentional attack |                       1 |              nan |                  nan |  5.45712e+06 | 2014-05-11 18:38:00 | 2014-05-11 18:39:00 | 0 days 00:01:00 |                    0.02 |            1 |
| Minnesota  | MRO           | East North Central |            -1.5 | cold               | severe weather     |                    3000 |              nan |                70000 |  5.3109e+06  | 2010-10-26 20:00:00 | 2010-10-28 22:00:00 | 2 days 02:00:00 |                   50    |            1 |
| Minnesota  | MRO           | East North Central |            -0.1 | normal             | severe weather     |                    2550 |              nan |                68200 |  5.38044e+06 | 2012-06-19 04:30:00 | 2012-06-20 23:00:00 | 1 days 18:30:00 |                   42.5  |            1 |
| Minnesota  | MRO           | East North Central |             1.2 | warm               | severe weather     |                    1740 |              250 |               250000 |  5.48959e+06 | 2015-07-18 02:00:00 | 2015-07-19 07:00:00 | 1 days 05:00:00 |                   29    |            1 |

We now create our permutation test to see if the model performs better or worse for landlocked states. Our null and alternative hypotheses will be as follows:

- Null: The final model is fair. Its F-1 score for both landlocked states and not landlocked states are roughly the same and any difference is due to chance.
- Alternative: The final model is unfair. The F-1 score for landlocked states and not landlocked states are not equal.

We will repeatedly shuffle the `landlocked` column, split the data into landlocked and not landlocked, and conduct a permutation test based on the absolute difference between the F-1 scores for each group. We will use a p-value of 0.05. The following figure shows the distribution of absolute differences between F-1 scores for the landlocked and not landlocked categories

The p-value calculated, which is the probability of a statistic as or more extreme as my test statistic occurring in the wild, is 0.508. Therefore, we fail to reject our null hypothesis that the final model is fair. It is most likely, based on the data and tests, that our final model does not find a distinction when grouping based off the landlocked status of the state in which the power outage occurred.

