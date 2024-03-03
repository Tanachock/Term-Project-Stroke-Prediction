# Stroke-Prediction
Import Datasets
```python
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
```
## Preprocessing<br>
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

