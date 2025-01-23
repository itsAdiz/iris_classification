#{{{IMPORTS}}} 

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#{{{Step 1 : Data Preparation  }}}


# loading and prepare the dataset
dataset_src = "./dataset/Iris.csv"
iris_data = pd.read_csv(dataset_src)

# dropping the Id column 
iris_data = iris_data.drop('Id', axis=1)

# defining feature columns
feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
print(f"Features being used: {feature_columns}")


# {{{ STEP 2-  Data Processing }}}

# preparing features (X) and target (y)
X = iris_data[feature_columns]
y = iris_data['Species']

# creating mapping for species
species_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y = y.map(species_mapping)

# {{{ Scaler , data spliting   }}}

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3,random_state=66)

# {{{ MODEL TRAINING }}}
# as we have to test multiple algorithms in order to find a best optimal algorithm 
# so we can proceed with with a dict of models names and their algorithm constructors 
models = {
    "Logistic Regression": LogisticRegression(random_state=66),
    "Decision Tree": DecisionTreeClassifier(random_state=66),
    "Random Forest": RandomForestClassifier(random_state=66),
    "SVM": SVC(probability=True,random_state=66)
}
# tracking vars
best_model_name = None
best_model = None
best_accuracy = 0

# Step 6: Train and evaluate models
print("\nModel Evaluation Results:")
print("-" * 50)

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)

    #{{ Metrics }}
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel: {name}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Display confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Display classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Performing cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=8)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.4f}")
    
    #updating the mmodel if current one is better
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name
        best_model = model

# saving the models
print(f"\nBest Model: {best_model_name} with accuracy {best_accuracy:.4f}")

try:
    # Saving the optimial model
    joblib.dump(best_model, './model/model.pkl')
    print("model saved to ./model/model.pkl")

    # Saving the scaler
    joblib.dump(scaler, './model/scaler.pkl')
    print("Scaler saved to ./model/scaler.pkl")
except Exception as e:
    print(f"error saving model: {e}")
