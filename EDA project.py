import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("C:/Users/USER/Dev/cfehome/train_2.csv")
#claim is the target and other cols are features, Y=claim x=features in equations, x is is claim in graphs and y is features

infos=df.info()
bottpm=df.tail()
top=df.head()
detail=df.describe()
#checking for errors
Na=df.isna().sum()
Null=df.isnull().sum()
duplicate=df.duplicated().sum()
cols=df.columns
#DataTypes=df.dtypes()
#checking class imbalancing at target column isclaim
classImbalancing=df.groupby("is_claim").count()
classImbalancing #the output shows large variation in data set
#viewing target variable destribution
sns.countplot(x=df["is_claim"], data=df)
plt.show()
#removing the unique column from the data set, the feature not helping to achieve target
df1=df.drop("policy_id", axis=1)
#boxplot of all the figures
df1.boxplot()
plt.xticks(rotation=45)
#plt.ylim(0, 2)
plt.title("Box plot of All columns")
plt.show()
#as we notice some outliers in data: from garph

#is_claim>0.8 but boolean values
#policyholder>0.73
#gearbox>5.5
#ageOfcar>0.5
#populationDensity>60000

#Now removing outliers of the population density
quant75=df1["population_density"].quantile(0.75)
quant25=df1["population_density"].quantile(0.25)
IqrPopDensity=quant75-quant25
#print(IqrPopDensity)
#upper/lower Threshold
Upper=quant75 + (1.5*IqrPopDensity)
Lower=quant25 - (1.5*IqrPopDensity)
#by susbsetting our data we are finding values outside of these limits
df1=df1[(df1["population_density"] > Lower) & (df["population_density"] < Upper)]
#print(df1["population_density"].describe())

sns.boxplot(data=df1, x="is_claim", y="age_of_car")
plt.show()
print(df1["age_of_car"].describe())
#cleaning age of car
quant75age=df1["age_of_car"].quantile(0.75)
quant25age=df1["age_of_car"].quantile(0.25)
Iqr=quant75age-quant25age
#print(Iqr)
#upper/lower Threshold
UpperAge=quant75age + (1.5*Iqr)
LowerAge=quant25age - (1.5*Iqr)
#by susbsetting our data we are finding values outside of these limits
df1=df1[(df1["age_of_car"] > LowerAge) & (df["age_of_car"] < UpperAge)]
#print(df1["age_of_car"].describe())

#Age of policy holders clean, age_of_policyholder
quant75Holder=df1["age_of_policyholder"].quantile(0.75)
quant25Holder=df1["age_of_policyholder"].quantile(0.25)
IqrHolder=quant75Holder-quant25Holder
#print(IqrHolder)
#upper/lower Threshold
UpperHolder=quant75Holder + (1.5*IqrHolder)
LowerHolder=quant25Holder - (1.5*IqrHolder)
#by susbsetting our data we are finding values outside of these limits
df1=df1[(df1["age_of_policyholder"] > LowerHolder) & (df["age_of_policyholder"] < UpperHolder)]
#print(df1["age_of_policyholder"].describe())

#now again box plot
df1.boxplot()
plt.title("clean data#1")
plt.xticks(rotation=45)
plt.show()

#cleaned the main useful data
#correlation finding so that only considering the relavent columns and removing irrelevant ones, correlation with isClaim
correlation=df1.corr()
print(correlation)
plt.figure(figsize=(17, 17))
sns.heatmap(correlation, annot=True)
plt.title("Correlation Heatmap of the data")
plt.show()
#now dropping make, height, gearbox, air bags column
df1=df1.drop(["height", "gear_box", "make", "airbags"], axis=1)
#print(df1.describe())

#pair plotting data hue as is_claim so that points of data in graph divided into class0 and class1
#mostly the last column of your data is your target variable
sns.pairplot(data=df1, hue="is_claim")
plt.title("pair plot of is_claim")
plt.show()

#for any unique values
uniques=df1["is_claim"].unique
#print(uniques)
#Extracting categorical columns by using for loop the columns having strings and object values separating them basically
CatFeatures=[col for col in df1.columns if col in df1.select_dtypes(include=object).columns]
#print(CatFeatures)
#Now saving all feature columns in features, as we know that features are all the columns that are not target variable
features=[col for col in df1.columns if col not in ["is_claim"]]
#print(features)

#making X, y Coordinate of all the features and target variable and putting all the values in it
X, y=df1.loc[:, features], df1.loc[:, "is_claim"]
#print(X.shape)
#Now using label encoder to make the object/string data to the boolean or integer or numeric
from sklearn.preprocessing import LabelEncoder
#MAKING isinstance
labelencode=LabelEncoder()
#changing categorical data iterating for loop over each CatFeatures
for col in CatFeatures:
    #storing its numeric values
    X[col]=labelencode.fit_transform(df1[col])
    #will access df1 columns 1 by 1 and store in X[col]
print("This is the encoded output: {} ".format(X))

#now the model will be furthr breaken into train and test


