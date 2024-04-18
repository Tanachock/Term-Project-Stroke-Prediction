# About the project
The objective of doing this project is Applying machine learning knowledge to real-world data To study and follow up and modify the model to suit the data and provide the most appropriate results

> We presume that you have some understanding about basic ML and their fundamentals, so we may proceed fast and discuss only the crucial topics.
![](https://media1.tenor.com/m/dKxQKwSGHMAAAAAd/stroke-cool.gif)
# Let have a look at ours dataset!
This dataset comes from [Kaggle](https://www.kaggle.com/datasets/imoore/60k-stack-overflow-questions-with-quality-rate), which includes healthcare information on many various groups of individuals, regardless of gender, age, weight, and so on, then we carefully examine them and put them into different models to make predictions that have the best outcome.

and for the 0 phase of our journey let's import some important library to use in our project


## First phase data review (to get some insight)
Let's go on to the first phase, where we import the dataset using this command.

```python
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
```
If you want to know what the data looks like. Use this command.
```python
df.head()
```

It should show like this 
![image](https://github.com/Tanachock/Term-Project-Stroke-Prediction/assets/83536257/38725ab9-b9b2-41e6-8ce2-329a20b196ca)

Now that we know which characteristic (column) of these data look like, some are numerical and some are categorical, we can know what to do with them.

We know that our dataset includes 12 features, and **Stroke** feature is our target class. Let's take a closer look at them.

![output](https://github.com/Tanachock/Term-Project-Stroke-Prediction/assets/83536257/9fbd3ca9-9254-4f9d-b5d7-799bc095097f)

oopsie! That's bad since we now have **imbalanced dataset**, which is a significant problem here because when we train the model, it will notice just one class more than another, making our model biased and only predicted to that class, but don't worry, we're going to fix that later.

What about regular features? Is there something wrong like the target class? Let's check my locating missing values first.'
```
id                     0
gender                 0
age                    0
hypertension           0
heart_disease          0
ever_married           0
work_type              0
Residence_type         0
avg_glucose_level      0
bmi                  201
smoking_status         0
stroke                 0
```

Again, we have another problem with an imbalanced dataset and a missing value, but it is not as serious as the previous one, so we will fix both of them.


### some insight from the first phase
- we know that we have 5110 sample with 201 missing value in BMI feature
- significant imbalanced dataset 

> there is no duplicate data or any other problem in our dataset XD let move on

## Second phase Preprocessing<br>

### Handling Missing Values
> This process are easy than imbalanced one so we fix it first

we know that BMI feature is missing 201 rows so we fix it by fill the missing with average of bmi 

```python
# handling missing values
avg = df['bmi'].mean()

df.bmi = (df.bmi.fillna(28.89))
print("AVG of this Adult BMI values = ", avg)
```
with the new BMI value around 28.89 fill 201 value so it not much to impact performance of dataset

### Now with imbalanced problem

there are many ways to solve this problem but we use SMOTE, to put it simply SMOTE create new sample by learning from dataset to make minor class match the major class
```python
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
```
we already seperate target and normal feature to X and y before we used smote

the rest of preprocess is encoded data into number and delete some least importance feature

## Create the ML model
we spilt train and test by 60/40 (to make sure that it is not overfit because we used smote)

we have create model:
- randomforest
- logistic regression
- KNN
- decisiontree
- svm

while all of this model has high accuracy but we not decide using only that score so we use confusion matrix and f1 score to find what best

while avg score of each model are the same (around 0.8 acuracy and f1 around 0.85)

but random forest model has the best result with accuracy 0.969 and f1 score 0.968 but it just a number let have a closer look
```python
rf1 =RandomForestClassifier(class_weight='balanced' , random_state=42, n_estimators = 100 , max_depth=13)
rf1.fit(x_train,y_train)
```
because RF(randomforest) can handle imbalanced dataset better than other model we made , and after tuing some parameter we the best possible result with RF model

![image](https://github.com/Tanachock/Term-Project-Stroke-Prediction/assets/83536257/9a4229fd-84ad-4e36-80b3-9c7f1b104df0)

this is confusion matrix of RF model it showed that model performed well and predict most case correct but has some minor false case


## Summary
- Major problem is imbalanced dataset but we solve it but using smote to make minor has more sample
- there is missing value problem but not significant we solve it by fill it with average of it feature
- best possible model is Randomforest but we think it not truly the best because there is many method to tuning the model to perform better
- keep in mind that even though model perform very good we cannot use this model to predict any real stroke case because human has more complex system than 12 feature in our dataset

> sorry if it too fast for you because this is our first ever big project and we try the best to explaining it XD

## Ref
[Our dataset](https://www.kaggle.com/datasets/imoore/60k-stack-overflow-questions-with-quality-rate)

[SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
