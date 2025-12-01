import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import pickle

'''
What this does:

pandas → to handle your dataset (tables).

train_test_split → to split data into train and test.

DecisionTreeClassifier → our ML model.

accuracy_score → to check how good the model is.

Now load your CSV:
'''
df = pd.read_csv("G:\csp project\ml projectsmaple\Crop_recommendation.csv")
# print(df) # we can used to see the tables in a terminal formate in the google coolabe "df" we see in a table formate
print(df.head()) # it can print the first row

# ==================Step 2 – Split into features (X) and target (y)===

# x=df.drop(columns=["lable"])  # it can remove the lable data temporary if we want permantly we use the (inplace=True)


# X = input features (what we use to predict)
X = df.drop(columns=['label']) 
# print(X) # it can remove the lable data

# y = target/output (what we want to predict)
y = df['label'] # it is used for the predict the value
# print(y) # it can print the values in the terminal


# ==================== print(y.unique()) ================
'''
['rice' 'maize' 'chickpea' 'kidneybeans' 'pigeonpeas' 'mothbeans'
 'mungbean' 'blackgram' 'lentil' 'pomegranate' 'banana' 'mango' 'grapes'
 'watermelon' 'muskmelon' 'apple' 'orange' 'papaya' 'coconut' 'cotton'
 'jute' 'coffee']
'''

print(X.head())
print(y.head())


# ====================== 🥉 Step 3 – Train/Test split =======================
# for more matter refer the notes.txt
'''
We must check if our model works on unseen data.
So we split data into:

Train set → used to teach the model.

Test set → used only to evaluate.
'''

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% test, 80% train
    random_state=42,     # for reproducible results
    stratify=y           # keep class balance
)

print("Train size:", X_train.shape) # it can tell the how many rows adn columns Train size: (1760, 7)
print("Test size :", X_test.shape) # Test size : (440, 7)


# Create the model
dt_model = DecisionTreeClassifier(
    random_state=42,     # same result every run
    max_depth=None,      # tree can grow fully (you can tune later)
)

# Train (fit) the model on training data
dt_model.fit(X_train, y_train)

# Save with joblib
joblib.dump(dt_model, 'dt_model_joblib.pkl')
# ============ dump =======
'''
dumps() = dump to string
It converts a Python object into binary string instead of file.
'''
# Save with pickle
with open('dt_model_pickle.pkl', 'wb') as f:
    pickle.dump(dt_model, f)
    


# Predict crops for the test set
y_pred = dt_model.predict(X_test) #X_text contained the only the x_feactures 

# Check a few predictions vs actual
print("Predicted:", y_pred[:10])
print("Actual   :", y_test.values[:10])
    

acc = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {acc:.4f}")


# ==========================🌾 Step 7 – Try a manual sample (predict for custom input)
'''
Let’s say you want to know which crop to grow for:

N = 90

P = 42

K = 43

temperature = 20.8

humidity = 82

ph = 6.5

rainfall = 200

This is how you predict:
'''

import numpy as np

# Sample input: [N, P, K, temperature, humidity, ph, rainfall]
sample = np.array([[90, 42, 43, 20.8, 82.0, 6.5, 200.0]])

predicted_crop = dt_model.predict(sample)
print("Recommended crop:", predicted_crop[0])

