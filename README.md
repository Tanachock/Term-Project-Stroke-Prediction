# About the project
The purpose of this project is to implement machine learning to real world scnario XD

> We presume that you have some understanding about basic ML and their fundamentals, so we may proceed fast and discuss only the crucial topics.

# Let have a look at ours dataset!
This dataset comes from [Kaggle](https://www.kaggle.com/datasets/imoore/60k-stack-overflow-questions-with-quality-rate), which includes healthcare information on many various groups of individuals, regardless of gender, age, weight, and so on, then we carefully examine them and put them into different models to make predictions that have the best outcome.

and for the 0 phase of our journey let's import some important library to use in our project

```python
#import things
```

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
```python
#explore target data
y = df['stroke']
yf =  pd.DataFrame(y)
print("number of patients that not have a stroke = " , y.value_counts()[0])
print("number of patients that have a stroke = " , y.value_counts()[1])

sns.countplot(x= y)
plt.xlabel("0 represents no strokes and 1 represents strokes")
plt.ylabel("counting ")
plt.title('Comparison between patients with strokes and without stroke')
```




## Second phase Preprocessing<br>
### Check Missing value
```python
df.isnull().sum() #check missing value
```
Output
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
Found a missing value namely BMI value.
### Handling Missing Values
Plot graph to see the frequency range of BMI values.<br><br>
![Screenshot 2567-03-03 at 03 01 08](https://github.com/Tanachock/Stroke-Prediction/assets/160312026/8cc04d35-3231-4119-947c-46c154e6be3b)<br>

The frequency of BMI values ​​is between 20 and 40.<br>
Use the mean instead of missing values.<br>
```python
avg = df['bmi'].mean()

df.bmi = (df.bmi.fillna(28.89))
print("AVG of this Adult BMI values = ", avg)
```
Output
```
28.89
```
Check target class<br><br>
![Screenshot 2567-03-03 at 03 16 17](https://github.com/Tanachock/Stroke-Prediction/assets/160312026/03ada2af-d5ab-4378-b99e-adf82d90dd18)<br>
0 means it indicates a stroke.<br>
1 means it's not a stroke.<br>

Datasets imbalance<br>
### Handle imbalance dataset
Use SMOTE to solve the imbalance dataset.<br>
SMOTE(Synthetic Minority Oversampling Technique) It is adds data to small classes by synthesizing new data from the original data set.
```python
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
```
After use SMOTE<br><br>
![Screenshot 2567-03-03 at 03 29 52](https://github.com/Tanachock/Stroke-Prediction/assets/160312026/898f41d3-59a2-43e2-9ef7-644d5deeaad2)<br>

