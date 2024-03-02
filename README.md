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
Plot graph to see the frequency range of BMI values.
