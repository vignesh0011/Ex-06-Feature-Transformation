# Ex-06-Feature-Transformation

## AIM

To read the given data and perform Feature Transformation process and save the data to a file.

## EXPLANATION

Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## ALGORITHM

### STEP 1:
Read the given Data

### STEP 2:
Clean the Data Set using Data Cleaning Process

### STEP 3:
Apply Feature Transformation techniques to all the features of the data set

### STEP 4:
Print the transformed features

## PROGRAM:
```
NAME: M VIGNESH
REG.NO:212220233002
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv("data_trans.csv")
df

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()

df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['HighlyNegativeSkew']=np.log(df.HighlyNegativeSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['ModeratePositiveSkew_1'], parameters=stats.yeojohnson(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()

df['ModerateNegativeSkew_1'], parameters=stats.yeojohnson(df.ModerateNegativeSkew)
sm.qqplot(df.ModerateNegativeSkew_1,fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))

sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

df2=df.copy()

df2['HighlyPositiveSkew']= 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()
```

## OUTPUT:
![image](https://user-images.githubusercontent.com/94154683/197689770-3a3eaeb0-893e-44c8-8f68-44e32a2649b0.png)

![image](https://user-images.githubusercontent.com/94154683/197689814-3694160d-908f-4231-a0db-650ff010af04.png)

![image](https://user-images.githubusercontent.com/94154683/197689850-6e468f71-c5b0-47b8-ac0d-b867332e873f.png)

![image](https://user-images.githubusercontent.com/94154683/197689891-94aeffa0-8937-455c-871e-6c7ddfa8ff51.png)

![image](https://user-images.githubusercontent.com/94154683/197689932-daa0397d-917c-40e0-859b-62ce48cd7d8f.png)

![image](https://user-images.githubusercontent.com/94154683/197689974-df9a8993-a064-4a73-9564-0e9bb3b7d61e.png)

![image](https://user-images.githubusercontent.com/94154683/197690077-9970da20-7ff7-4b8b-bf25-0aa8478de4ea.png)

![image](https://user-images.githubusercontent.com/94154683/197690168-8e6c3d09-2046-4f34-b404-2f3a55de7781.png)

![image](https://user-images.githubusercontent.com/94154683/197690206-127a5a92-b62f-4456-9d54-61b3fc83e642.png)

![image](https://user-images.githubusercontent.com/94154683/197690242-c3a3e2e2-9150-45ba-a9c3-5fb35ad47407.png)


## RESULT:
Thus feature transformation is done for the given dataset.
