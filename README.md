# Ex.No: 13 Miniproject 
### DATE:  22-04-2024                                                                      
### REGISTER NUMBER : 212221040146
### AIM: 
To write a program to train the classifier for -----------------.
###  Algorithm:
Step:1 Import package
Step:2 Get the data
Step:3 Split the data
Step:4 Scale the data
Step:5 Instantiate model
Step 6: Create Gradio Function
Step 7: Print Result

### Program:

```
#import packages
import numpy as np
import pandas as pd

pip install gradio

pip install typing-extensions --upgrade

!python --version

pip install --upgrade typing

import gradio as gr

import pandas as pd

#get the data
data = pd.read_csv('diabetes.csv')
data.head()
```

![image](https://github.com/santhoshkumar24263/miniproject/assets/127171952/75cd74ce-9fcd-4c9c-875f-a6420e3af9dc)

```

print(data.columns)

x = data.drop(['Outcome'], axis=1)
y = data['Outcome']
print(x[:5])

```
![image](https://github.com/santhoshkumar24263/miniproject/assets/127171952/4b687d05-d456-4441-9d20-3cca77d59c79)

```

#split data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(x,y)

#scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

instatiate model
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(max_iter=1000, alpha=1)
model.fit(x_train, y_train)
print("Model Accuracy on training set:", model.score(x_train, y_train))
print("Model Accuracy on Test Set:", model.score(x_test, y_test))

print(data.columns)

#create a function for gradio
def diabetes(Pregnancies, Glucose, Blood_Pressure, SkinThickness, Insulin, BMI,Diabetes_Pedigree, Age):
    x = np.array([Pregnancies,Glucose,Blood_Pressure,SkinThickness,Insulin,BMI,Diabetes_Pedigree,Age])
    prediction = model.predict(x.reshape(1, -1))
    if(prediction==0):
      return "NO"
    else:
      return "YES"

outputs = gr.Textbox()
app = gr.Interface(fn=diabetes, inputs=['number','number','number','number','number','number','number','number'], outputs=outputs,description="Detection of Diabeties")
app.launch(share=True)
```

### Output:

![Screenshot 2024-05-06 111216](https://github.com/santhoshkumar24263/miniproject/assets/127171952/53bb1e31-5bad-48c9-8baa-71b72830b8f3)

![Screenshot 2024-05-06 111246](https://github.com/santhoshkumar24263/miniproject/assets/127171952/8f4118b8-92fb-4ee4-a1c4-8d25758e2f6d)

![Screenshot 2024-05-06 111308](https://github.com/santhoshkumar24263/miniproject/assets/127171952/1a66369a-b4ed-4a47-9e2c-480821c90b0e)

### Result:
Thus the system was trained successfully and the prediction was carried out.
