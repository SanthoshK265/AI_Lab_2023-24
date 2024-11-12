
# Ex.No: 13 Learning – Use Supervised Learning(Miniproject)
## DATE:     
04.11.2024
## REGISTER NUMBER : 
212222060222
## AIM: 
To write a program to train the classifier for Weather.
##  Algorithm:
Sure, here is the algorithm in the specified format:

**Algorithm for Student Performance Prediction:**

Step 1: Start the program.
Step 2: Import necessary packages, such as NumPy and Pandas.
Step 3: Install and import Gradio for creating the user interface.
Step 4: Load the Student Performance dataset using Pandas.
Step 5: Split the dataset into input features (`X`) representing factors affecting performance (e.g., study hours, attendance, past grades) and target labels (`y`) representing the performance outcome (such as pass/fail or grade).
Step 6: Split the data into training and testing sets using `train_test_split`.
Step 7: Standardize the training and testing data using `StandardScaler`.
Step 8: Instantiate the `MLPClassifier` model with 1000 iterations and train the model on the training data.
Step 9: Print the model's accuracy on both the training and testing sets.
Step 10: Take input values for student features (e.g., study hours, attendance) and predict the performance outcome using the trained model.
Step 11: Stop the program.
## Program:
STUDENT PERFORMANCE
```
import smtplib
from matplotlib import style
import seaborn as sns
sns.set(style='ticks', palette='RdBu')
#sns.set(style='ticks', palette='Set2')
import pandas as pd
import numpy as np
import time
import datetime 
%matplotlib inline
import matplotlib.pyplot as plt
from subprocess import check_output
pd.options.display.max_colwidth = 1000
from time import gmtime, strftime
Time_now = strftime("%Y-%m-%d %H:%M:%S", gmtime())
import timeit
start = timeit.default_timer()
pd.options.display.max_rows = 100

from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV, SelectKBest
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn import svm
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
```
Read the data
```
data = pd.read_csv('../input/xAPI-Edu-Data.csv')
df = data
```
Describe the data

1
```
data.columns
```
2
```
data.head(n=2).T
```
3
```
data.describe()
```
Categorical features
```
categorical_features = (data.select_dtypes(include=['object']).columns.values)
categorical_features
```
Numerical Features
```
numerical_features = data.select_dtypes(include = ['float64', 'int64']).columns.values
numerical_features
```
Pivot tables
```
pivot = pd.pivot_table(df,
            values = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion'],
            index = ['gender', 'NationalITy', 'PlaceofBirth'], 
                       columns= ['ParentschoolSatisfaction'],
                       aggfunc=[np.mean], 
                       margins=True).fillna('')
pivot
```
```
pivot = pd.pivot_table(df,
            values = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion'],
            index = ['gender', 'NationalITy', 'PlaceofBirth'], 
                       columns= ['ParentschoolSatisfaction'],
                       aggfunc=[np.mean, np.std], 
                       margins=True)
cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)
plt.subplots(figsize = (30, 20))
sns.heatmap(pivot,linewidths=0.2,square=True )
```
Simple plots
Correlations
```
def heat_map(corrs_mat):
    sns.set(style="white")
    f, ax = plt.subplots(figsize=(20, 20))
    mask = np.zeros_like(corrs_mat, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True 
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corrs_mat, mask=mask, cmap=cmap, ax=ax)

variable_correlations = df.corr()
#variable_correlations
heat_map(variable_correlations)
```
```
df_small = df[['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion', 'NationalITy']]
sns.pairplot(df_small, hue='NationalITy')
```
```
df.columns
```

Complex plots
Modify the original dataframe itself to make variables as numbers
```
mod_df = df 

gender_map = {'M':1, 
              'F':2}

NationalITy_map = {  'Iran': 1,
                     'SaudiArabia': 2,
                     'USA': 3,
                     'Egypt': 4,
                     'Lybia': 5,
                     'lebanon': 6,
                     'Morocco': 7,
                     'Jordan': 8,
                     'Palestine': 9,
                     'Syria': 10,
                     'Tunis': 11,
                     'KW': 12,
                     'KuwaIT': 12,
                     'Iraq': 13,
                     'venzuela': 14}
PlaceofBirth_map =  {'Iran': 1,
                     'SaudiArabia': 2,
                     'USA': 3,
                     'Egypt': 4,
                     'Lybia': 5,
                     'lebanon': 6,
                     'Morocco': 7,
                     'Jordan': 8,
                     'Palestine': 9,
                     'Syria': 10,
                     'Tunis': 11,
                     'KW': 12,
                     'KuwaIT': 12,
                     'Iraq': 13,
                     'venzuela': 14}

StageID_map = {'HighSchool':1, 
               'lowerlevel':2, 
               'MiddleSchool':3}

GradeID_map =   {'G-02':2,
                 'G-08':8,
                 'G-09':9,
                 'G-04':4,
                 'G-05':5,
                 'G-06':6,
                 'G-07':7,
                 'G-12':12,
                 'G-11':11,
                 'G-10':10}

SectionID_map = {'A':1, 
                 'C':2, 
                 'B':3}

Topic_map  =    {'Biology' : 1,
                 'Geology' : 2,
                 'Quran' : 3,
                 'Science' : 4,
                 'Spanish' : 5,
                 'IT' : 6,
                 'French' : 7,
                 'English' :8,
                 'Arabic' :9,
                 'Chemistry' :10,
                 'Math' :11,
                 'History' : 12}
Semester_map = {'S':1, 
                'F':2}

Relation_map = {'Mum':2, 
                'Father':1} 

ParentAnsweringSurvey_map = {'Yes':1,
                             'No':0}

ParentschoolSatisfaction_map = {'Bad':0,
                                'Good':1}

StudentAbsenceDays_map = {'Under-7':0,
                          'Above-7':1}

Class_map = {'H':10,
             'M':5,
             'L':2}

mod_df.gender  = mod_df.gender.map(gender_map)
mod_df.NationalITy     = mod_df.NationalITy.map(NationalITy_map)
mod_df.PlaceofBirth     = mod_df.PlaceofBirth.map(PlaceofBirth_map)
mod_df.StageID       = mod_df.StageID.map(StageID_map)
mod_df.GradeID = mod_df.GradeID.map(GradeID_map)
mod_df.SectionID    = mod_df.SectionID.map(SectionID_map)
mod_df.Topic     = mod_df.Topic.map(Topic_map)
mod_df.Semester   = mod_df.Semester.map(Semester_map)
mod_df.Relation   = mod_df.Relation.map(Relation_map)
mod_df.ParentAnsweringSurvey   = mod_df.ParentAnsweringSurvey.map(ParentAnsweringSurvey_map)
mod_df.ParentschoolSatisfaction   = mod_df.ParentschoolSatisfaction.map(ParentschoolSatisfaction_map)
mod_df.StudentAbsenceDays   = mod_df.StudentAbsenceDays.map(StudentAbsenceDays_map)
mod_df.Class  = mod_df.Class.map(Class_map)
#mod_df.to_csv(path + 'mod_df.csv')
```
#data = df
sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(4, 4, figsize=(20,20))
sns.despine(left=True)
sns.distplot(df['NationalITy'],  kde=False, color="b", ax=axes[0, 0])
sns.distplot(df['PlaceofBirth'],        kde=False, color="b", ax=axes[0, 1])
sns.distplot(df['StageID'],        kde=False, color="b", ax=axes[0, 2])
sns.distplot(df['GradeID'],        kde=False, color="b", ax=axes[0, 3])
sns.distplot(df['SectionID'], kde=False, color="b", ax=axes[1, 0])
sns.distplot(df['Topic'],  kde=False, color="b", ax=axes[1, 1])
sns.distplot(df['Relation'],     kde=False, color="b", ax=axes[1, 2])
sns.distplot(df['raisedhands'],  kde=False, color="b", ax=axes[1, 3])
sns.distplot(df['VisITedResources'],      kde=False, color="b", ax=axes[2, 0])
sns.distplot(df['AnnouncementsView'],      kde=False, color="b", ax=axes[2, 1])
sns.distplot(df['Discussion'],    kde=False, color="b", ax=axes[2, 2])
sns.distplot(df['ParentAnsweringSurvey'],    kde=False, color="b", ax=axes[2, 3])
sns.distplot(df['ParentschoolSatisfaction'],kde=False, color="b", ax=axes[3, 0])
sns.distplot(df['StudentAbsenceDays'],       kde=False, color="b", ax=axes[3, 1])
sns.distplot(df['Class'],      kde=False, color="b", ax=axes[3, 2])
#sns.distplot(df['Fedu'],      kde=False, color="b", ax=axes[3, 3])
plt.tight_layout()
```
categorical_features = (mod_df.select_dtypes(include=['object']).columns.values)
categorical_features
```
```
mod_df_variable_correlations = mod_df.corr()
#variable_correlations
heat_map(mod_df_variable_correlations)
```

```
df.columns
```

```
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
#import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn import svm

df_copy = pd.get_dummies(mod_df)

df1 = df_copy
y = np.asarray(df1['ParentschoolSatisfaction'], dtype="|S6")
df1 = df1.drop(['ParentschoolSatisfaction'],axis=1)
X = df1.values
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.50)

radm = RandomForestClassifier()
radm.fit(Xtrain, ytrain)

clf = radm
indices = np.argsort(radm.feature_importances_)[::-1]

# Print the feature ranking
print('Feature ranking:')

for f in range(df1.shape[1]):
    print('%d. feature %d %s (%f)' % (f+1 , 
                                      indices[f], 
                                      df1.columns[indices[f]], 
                                      radm.feature_importances_[indices[f]]))
```

```
temp = pd.DataFrame(allscores, columns=['classifier', 'score'])
#sns.violinplot('classifier', 'score', data=temp, inner=None, linewidth=0.3)
plt.figure(figsize=(15,10))
sns.factorplot(x='classifier', 
               y="score",
               data=temp, 
               saturation=1, 
               kind="box", 
               ci=None, 
               aspect=1, 
               linewidth=1, 
               size = 10)     
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
```

## Output:
![WhatsApp Image 2024-11-12 at 08 56 13_5d093948](https://github.com/user-attachments/assets/86d1d668-b886-44ea-86e9-3af763b565da)
![WhatsApp Image 2024-11-12 at 08 56 13_73ec8182](https://github.com/user-attachments/assets/e3178f33-0dc3-4fc0-a801-a468f5de3e8d)
![WhatsApp Image 2024-11-12 at 08 56 13_0273b9c6](https://github.com/user-attachments/assets/96ca366c-ccc3-42d4-a2c7-adcef4d9911a)
![WhatsApp Image 2024-11-12 at 08 56 13_5972a53a](https://github.com/user-attachments/assets/49bbcf4e-c630-474d-978a-b91395462be6)
![WhatsApp Image 2024-11-12 at 08 56 13_7731d2fc](https://github.com/user-attachments/assets/20b1a736-b2d2-44e1-af41-af827603fedf)
![WhatsApp Image 2024-11-12 at 08 56 14_6549df9f](https://github.com/user-attachments/assets/3b064561-6761-4dd8-942d-6954250015ea)
![WhatsApp Image 2024-11-12 at 08 56 14_63378eec](https://github.com/user-attachments/assets/5be6b985-fb6f-4d67-99cc-8e895440f46d)
![WhatsApp Image 2024-11-12 at 08 56 14_b2cb79ad](https://github.com/user-attachments/assets/1e41877a-f130-44e1-b56d-db1293c2cbf9)
![WhatsApp Image 2024-11-12 at 08 56 14_f8d98c40](https://github.com/user-attachments/assets/1d18a344-2c3c-486f-aed9-68854edf7079)


## Result:
Thus the system was trained successfully and the prediction was carried out.
