# Titanic-EDA-Project

from google.colab import drive
drive.mount('/content/drive')

#Importing Libraries and exploring data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
df = pd.read_csv('/content/drive/MyDrive/traintitanic.csv')
df.head()

df.info()

df.describe()

#Duplicate Values

df.duplicated().sum()

df.isnull().sum()

df[df.Age.isnull()]

round(df.isna().mean() * 100 ,2)

df['Embarked'].describe()

df['Embarked'].value_counts()

df['Embarked'].mode()

df['Embarked'].mode()[0]

df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

df.isnull().sum()

df[df['Sex'] == 'female'].Age.median()

df[df['Sex'] == 'male'].Age.median()

print(df[df['Pclass'] == 1].Age.median())
print(df[df['Pclass'] == 2].Age.median())
print(df[df['Pclass'] == 3].Age.median())

print(df[(df['Pclass'] == 1) & (df['Sex'] == 'female')].Age.median())
print(df[(df['Pclass'] == 1) & (df['Sex'] == 'male')].Age.median())
print(df[(df['Pclass'] == 2) & (df['Sex'] == 'female')].Age.median())
print(df[(df['Pclass'] == 2) & (df['Sex'] == 'male')].Age.median())
print(df[(df['Pclass'] == 3) & (df['Sex'] == 'female')].Age.median())
print(df[(df['Pclass'] == 3) & (df['Sex'] == 'male')].Age.median())

df.groupby(['Pclass', 'Sex'])['Age'].median()

def fill_age(pclass, sex):
    if pclass == 1:
        if sex == 'female':
            return 35
        else:
            return 40
    elif pclass == 2:
        if sex == 'female':
            return 28
        else:
            return 30
    else:
        if sex == 'female':
            return 21.5
        else:
            return 25
df['Age2'] = df['Age'].fillna(lambda x : fill_age(x['Pclass'], x['Sex']))

(df.Age2 == df.Age ).all()

df.isnull().sum()

df.head()

#Next steps:
#Drop Columns

df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

#Rename Columns and create new feature

df.columns = df.columns.str.lower()
df['family_size'] = df['sibsp'] + df['parch']

df.sample()

#Univariate Analysis

df.survived.value_counts()

df.survived.value_counts(normalize=True)

sns.countplot(x='survived', data=df);

df.pclass.value_counts(normalize=True)

df.sibsp.value_counts(normalize=True)

df.columns

def explore_categorical(df, col):
    print(f'### {col} ###')
    print(df[col].value_counts(normalize=True))
    sns.countplot(x=col, data=df);
    plt.show()
    
for col in ['pclass', 'sibsp', 'parch', 'embarked', 'sex', 'family_size','survived','age']:
    explore_categorical(df, col)

df.groupby('sibsp').survived.mean()

sns.barplot(x='sibsp', y='survived', data=df);

sns.barplot(x='sibsp', y='survived', data=df, ci=None);
plt.axhline(df.survived.mean(), color='black', linestyle='--');
plt.show()

#Bivariate Analysis

def survival_rate(df, col):
    print(df.groupby(col).survived.mean())
    sns.barplot(x=col, y='survived', data=df, ci=None);
    #plot horizontal line for overall survival rate
    plt.axhline(df.survived.mean(), color='black', linestyle='--')
    plt.show()
for col in ['pclass', 'sibsp', 'parch', 'embarked', 'sex', 'family_size','survived','age']:
    survival_rate(df, col)

df[df['family_size'] ==0].shape[0]

df[df['family_size'] ==0]['family_size'].count()

df['family_size'].value_counts()

df['family_size'].value_counts(normalize= True)

df[df['family_size'] ==0].survived.mean()

df.groupby('family_size').survived.mean().sort_values()

df.groupby('family_size').survived.mean().sort_values(ascending = False)

df.groupby('family_size').survived.mean().sort_values(ascending = False).head(3)

df.groupby('family_size').survived.mean().nlargest

df.groupby('family_size').survived.mean().nlargest(3)

sns.histplot(x='fare', data=df, hue='survived');

sns.histplot(x='fare', data=df, hue='survived', multiple='stack');

df_survived = df[df.survived == 1]
df_died = df[df.survived == 0]

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(x='fare', data=df_survived, ax=ax[0], kde=True, color='green')
sns.histplot(x='fare', data=df_died, ax=ax[1], kde=True, color='red')
ax[0].set_title('Survived')
ax[1].set_title('Died')
plt.show()

sns.histplot(x='age', data=df, hue='survived');

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(x='age', data=df_survived, ax=ax[0], kde=True, color='green')
sns.histplot(x='age', data=df_died, ax=ax[1], kde=True, color='red')
ax[0].set_title('Survived')
ax[1].set_title('Died')
plt.show()

df.describe()[['age', 'fare']]

df.describe()[['age', 'fare']]

sns.boxplot( x='fare', data=df);

#REMOVE OUTLIERS

def remove_outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    return df[(df[col] > lower_bound) & (df[col] < upper_bound)]

df = df[df.fare < 300]

sns.boxplot( x='fare', data=df);

df.fare.describe()

 df['age_group'] = pd.cut(df.age, bins=[0, 22, 27, 37, 82], labels=['child', 'young', 'adult', 'senior'])

df['fare_group'] = pd.cut(df.fare, bins=[-0.99, 8, 15, 35, 265], labels=['low', 'medium', 'high', 'very high'])

for col in ['age_group', 'fare_group']:
    survival_rate(df, col)

#Multivariate Analysis

sns.pairplot(df, hue='survived');

sns.barplot(x='pclass', y='survived', hue= 'sex', data=df, ci=None);

sns.barplot(x='embarked', y='survived', hue= 'sex', data=df, ci=None);

#Conclusion

fig, ax = plt.subplots(2, 4, figsize=(15, 8))
for i, col in enumerate(['pclass', 'sibsp', 'parch', 'family_size', 'age_group', 'fare_group', 'embarked', 'sex']):
    sns.barplot(x=col, y= 'survived', data=df, ci=None, ax=ax[i//4, i%4])
    ax[i//4, i%4].axhline(df.survived.mean(), color='black', linestyle='--')

plt.tight_layout()
plt.show()

female_df = df[df.sex == 'female']
male_df = df[df.sex == 'male']
female_df.survived.value_counts(normalize=True)

female_df.groupby('pclass').survived.mean()
pclass

sns.barplot(x='pclass', y='survived', data=female_df, ci=None);
plt.axhline(female_df.survived.mean(), color='black', linestyle='--')
plt.show()

male_df.survived.value_counts(normalize=True)
survived

male_df.groupby('pclass').survived.mean()

sns.barplot(x='pclass', y='survived', data=male_df, ci=None);
plt.axhline(male_df.survived.mean(), color='black', linestyle='--')

sex_class = pd.merge(female_df.groupby('pclass').survived.mean(), male_df.groupby('pclass').survived.mean(), on='pclass')
sex_class

sex_class.rename(columns= {'survived_x': 'female_survived', 'survived_y': 'male_survived'}, inplace=True)

df.groupby(['pclass', 'sex']).survived.mean()

df.groupby(['age_group', 'sex']).survived.mean()

pd.DataFrame(df.groupby(['age_group', 'sex']).survived.mean())

pd.DataFrame(df.groupby(['sex', 'age_group']).survived.mean())

age_sex = pd.DataFrame(df.groupby(['age_group', 'sex']).survived.mean()).sort_values(by='survived')
age_sex

age_sex = age_sex.reset_index()
age_sex

age_sex[age_sex['sex']== 'female'].iloc[0]

age_sex[age_sex['sex']== 'female'].iloc[0]['age_group']

age_sex[age_sex['sex']== 'female'].iloc[-1]

age_sex[age_sex['sex']== 'female'].iloc[-1]['age_group']

age_sex[age_sex['sex']== 'male'].iloc[0]['age_group']

age_sex[age_sex['sex']== 'male'].iloc[-1]['age_group']

df

df.drop(['sex', 'age_group', 'fare_group'], axis=1, inplace=True)

df.drop(['embarked'], axis=1, inplace=True)

df.drop(['age2'], axis=1, inplace=True)

df.corr()

df.corr()['survived']

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True);

#Insights

The higher the class, the higher the survival rate

The higher the fare, the higher the survival rate

Females had a higher survival rate bold text

