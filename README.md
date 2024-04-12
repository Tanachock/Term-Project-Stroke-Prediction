# About the project
The purpose of this project is to implement machine learning to real world scnario XD

> We presume that you have some understanding about basic ML and their fundamentals, so we may proceed fast and discuss only the crucial topics.

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

