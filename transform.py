import pandas as pd
import numpy as np

def transform_fda_data(raw_fda_results):
    if not raw_fda_results:
        return pd.DataFrame()
    
    extracted_records = []
    
    for record in raw_fda_results:
        patient = record.get('patient', {})
        age = patient.get('patientonsetage', np.nan)
        gender = patient.get('patientsex', 0)
        gender_map = {1: 'Male', 2: 'Female'}
        gender_str = gender_map.get(int(gender) if gender else 0, 'Unknown')
        reactions = patient.get('reaction', [])
        primary_reaction = reactions[0].get('reactionmeddrapt', 'Unknown') if reactions else 'Unknown'
        report_date = record.get('receiptdate', None)
        if report_date:
            report_date = pd.to_datetime(report_date, format='%Y%m%d', errors='coerce')
        raw_seriousness = record.get('serious', 2)
        risk_level = 1 if str(raw_seriousness) == '1' else 0
        
        age_val = float(age) if age and not pd.isna(age) else 40
        
        extracted_records.append({
            'report_date': report_date,
            'patient_age': age_val,
            'patient_gender': gender_str,
            'adverse_reaction': primary_reaction,
            'risk_level': risk_level 
        })
        
    df_fda = pd.DataFrame(extracted_records)
    df_fda['patient_age'] = df_fda['patient_age'].fillna(df_fda['patient_age'].mean())
    return df_fda

def transform_disease_data(raw_disease_results):
    """Disease.sh ke nested date-wise data ko daily metrics DataFrame mein convert karna"""
    if not raw_disease_results or 'cases' not in raw_disease_results:
        return pd.DataFrame()
    
    cases = raw_disease_results.get('cases', {})
    deaths = raw_disease_results.get('deaths', {})
    recoveries = raw_disease_results.get('recovered', {})
    dates = list(cases.keys())
    
    records = []
    for date in dates:
        records.append({
            'date': pd.to_datetime(date),
            'cumulative_cases': cases.get(date, 0),
            'cumulative_deaths': deaths.get(date, 0),
            'cumulative_recoveries': recoveries.get(date, 0)
        })
        
    df_disease = pd.DataFrame(records).sort_values('date').reset_index(drop=True)
    df_disease['daily_new_cases'] = df_disease['cumulative_cases'].diff().fillna(0).astype(int)
    df_disease['daily_new_deaths'] = df_disease['cumulative_deaths'].diff().fillna(0).astype(int)
    df_disease['daily_new_cases'] = df_disease['daily_new_cases'].clip(lower=0)
    df_disease['daily_new_deaths'] = df_disease['daily_new_deaths'].clip(lower=0)
    
    return df_disease


if __name__ == "__main__":
    from extract import fetch_openfda_data, fetch_disease_data
    
    print("Testing Transformation Layer...")
    raw_fda = fetch_openfda_data(limit=10)
    raw_disease = fetch_disease_data()
    
    print("\nTransforming Data...")
    df_fda_clean = transform_fda_data(raw_fda)
    df_disease_clean = transform_disease_data(raw_disease)
    
    print("\n--- FDA Cleaned Shape & Sample ---")
    print(df_fda_clean.shape)
    print(df_fda_clean.head(2))
    
    print("\n--- Disease Cleaned Shape & Sample ---")
    print(df_disease_clean.shape)
    print(df_disease_clean.head(2))