import os
import pickle
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    print("Error: Supabase Credentials missing!")
    supabase = None

def train_risk_model():
    print("Fetching training data from Supabase Cloud Database...")
    if supabase is None: return

    try:
        response = supabase.table("fda_adverse_events").select("*").execute()
        df = pd.DataFrame(response.data)
        
        if df.empty or len(df) < 10:
            print("Not enough data to train yet.")
            return
            
        print(f"Successfully fetched {len(df)} records for training.")
        reaction_counts = df['adverse_reaction'].value_counts()
        top_reactions = reaction_counts[reaction_counts >= 2].index
        df['adverse_reaction_cleaned'] = df['adverse_reaction'].apply(
            lambda x: x if x in top_reactions else 'OTHER'
        )
        
        X = df[['patient_age', 'patient_gender', 'adverse_reaction_cleaned']].copy()
        y = df['risk_level'].astype(int)
        
        num_neg = np.sum(y == 0)
        num_pos = np.sum(y == 1)
        scale_weight = (num_neg / num_pos) if num_pos > 0 else 1.0
        
        X = pd.get_dummies(X, columns=['patient_gender', 'adverse_reaction_cleaned'], drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print("Training Optimized XGBoost Classifier...")
        model = XGBClassifier(
            n_estimators=60,
            max_depth=3,
            learning_rate=0.03,
            min_child_weight=3, 
            gamma=1.0,           
            scale_pos_weight=scale_weight,
            random_state=42,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_preds) * 100
        test_acc = accuracy_score(y_test, test_preds) * 100
        
        print(f"\n======================================")
        print(f"📈 OVERFITTING / GAP ANALYSIS")
        print(f"======================================")
        print(f"Training Accuracy:   {train_acc:.2f}%")
        print(f"Validation Accuracy: {test_acc:.2f}%")
        print(f"Generalization Gap:  {abs(train_acc - test_acc):.2f}%")
        
        # 2. Precision, Recall & F1-Score
        print(f"\n======================================")
        print(f"📋 MEDICAL CLASSIFICATION REPORT")
        print(f"======================================")
        print(classification_report(y_test, test_preds, target_names=['Low Risk (0)', 'High Risk (1)']))
        
        # 3. Confusion Matrix Breakdown
        tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
        print(f"======================================")
        print(f"🧩 CONFUSION MATRIX BREAKDOWN")
        print(f"======================================")
        print(f"True Negatives (Correct Low Risk):  {tn}")
        print(f"False Positives (Type I Error):     {fp}")
        print(f"False Negatives (Type II Danger):   {fn}  <-- Critically important for healthcare!")
        print(f"True Positives (Correct High Risk): {tp}")
        print(f"======================================\n")
        model_metadata = {
            'model': model,
            'feature_columns': list(X.columns)
        }
        
        with open('healthcare_model.pkl', 'wb') as f:
            pickle.dump(model_metadata, f)
            
        print("XGBoost pipeline model and evaluation logging completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    train_risk_model()