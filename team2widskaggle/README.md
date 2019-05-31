# Team 2 TalkingData Kaggle competition
This directory contains the notebooks for Team 2's initial model to predict app downloads using data from TalkingData, China’s largest independent big data service platform. More information on this competition can be found [here](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection).

## Contents of folders
### EDA
* sample_data_EDA - EDA on sample data provided by TalkingData (100k clicks)
* train_vs_test_EDA - EDA on full train and test datasets

### feature_eng
* train_feature_engineering - code to create aggregate click count features for 2 and 3 feature groupings with annotations
* test_feature_engineering - code to create the same features for the test set

### train_predict
* creating_train_val_data - code to create training and validation datasets
* test_model_run - code to do practice run of model using lightGBM
* train_interactions - training model with interaction terms between features
* train_agg_counts - training model with features engineered in feature_eng folder. **Note: This script has some notes on lessons learned and next steps at the end**
