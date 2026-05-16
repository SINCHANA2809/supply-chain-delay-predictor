import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from xgboost import XGBClassifier
import joblib

# Load and prepare data
df = pd.read_csv('data/Train.csv')
df_model = df.copy()
df_model = df_model.drop('ID', axis=1)

le = LabelEncoder()
for col in ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']:
    df_model[col] = le.fit_transform(df_model[col])

df_model['weight_category'] = pd.cut(df_model['Weight_in_gms'],
                                      bins=[0, 2000, 4000, 10000],
                                      labels=[0, 1, 2]).astype(int)
df_model['discount_category'] = pd.cut(df_model['Discount_offered'],
                                        bins=[-1, 10, 30, 100],
                                        labels=[0, 1, 2]).astype(int)
df_model['cost_per_gram'] = (df_model['Cost_of_the_Product'] /
                              df_model['Weight_in_gms']).round(4)

X = df_model.drop('Reached.on.Time_Y.N', axis=1)
y = df_model['Reached.on.Time_Y.N']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Set MLflow experiment name
mlflow.set_experiment("supply-chain-delay-predictor")

# Define all models
models = {
    "Logistic_Regression": (LogisticRegression(random_state=42, max_iter=2000), True),
    "Random_Forest": (RandomForestClassifier(n_estimators=100, random_state=42), False),
    "XGBoost": (XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'), False),
    "Gradient_Boosting": (GradientBoostingClassifier(n_estimators=100, random_state=42), False),
}

# Train and log each model
for name, (model, needs_scaling) in models.items():
    with mlflow.start_run(run_name=name):
        
        Xtr = X_train_scaled if needs_scaling else X_train
        Xte = X_test_scaled if needs_scaling else X_test
        
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        
        mlflow.log_param("model_type", name)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall_delayed", recall_score(y_test, y_pred, pos_label=0))
        
        mlflow.sklearn.log_model(model, name)
        
        print(f"✅ {name} logged — Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# Log Voting Classifier separately
with mlflow.start_run(run_name="Voting_Classifier"):
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(random_state=42, max_iter=2000)
    
    voting_model = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
        voting='soft'
    )
    voting_model.fit(X_train, y_train)
    y_pred = voting_model.predict(X_test)
    
    mlflow.log_param("model_type", "Voting_Classifier")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall_delayed", recall_score(y_test, y_pred, pos_label=0))
    
    mlflow.sklearn.log_model(voting_model, "Voting_Classifier")
    
    print(f"✅ Voting_Classifier logged — Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

print("\n✅ All 5 experiments logged to MLflow!")
print("Run 'mlflow ui' in terminal to view the dashboard")