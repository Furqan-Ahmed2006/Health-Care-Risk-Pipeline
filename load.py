import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    print("Error: Supabase Credentials missing!")
    supabase = None

def load_fda_to_supabase(df_fda):
    if supabase is None or df_fda.empty:
        print("Skipping FDA Load.")
        return
    
    df_to_load = df_fda.copy()
    if 'report_date' in df_to_load.columns:
        df_to_load['report_date'] = df_to_load['report_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    initial_len = len(df_to_load)
    df_to_load = df_to_load.drop_duplicates(subset=["report_date", "patient_age", "adverse_reaction"])
    final_len = len(df_to_load)
    
    if initial_len != final_len:
        print(f"🧹 Cleaned {initial_len - final_len} duplicate records from the current incoming API batch.")
    
    records = df_to_load.to_dict(orient='records')
    
    try:
        response = supabase.table("fda_adverse_events").upsert(
            records,
            on_conflict="report_date,patient_age,adverse_reaction"
        ).execute()
        print(f"Loaded {len(records)} FDA records into Supabase (Cloud duplicates automatically skipped).")
    except Exception as e:
        print(f"Error loading FDA to Supabase: {e}")

def load_disease_to_supabase(df_disease):
    if supabase is None or df_disease.empty:
        print("Skipping Disease Load.")
        return
        
    df_to_load = df_disease.copy()
    if 'date' in df_to_load.columns:
        df_to_load['date'] = df_to_load['date'].dt.strftime('%Y-%m-%d')
    
    records = df_to_load.to_dict(orient='records')
    
    try:
        response = supabase.table("disease_daily_metrics").upsert(records, on_conflict="date").execute()
        print(f"Loaded/Updated {len(records)} disease records into Supabase.")
    except Exception as e:
        print(f"Error loading Disease to Supabase: {e}")

if __name__ == "__main__":
    from extract import fetch_openfda_data, fetch_disease_data
    from transform import transform_fda_data, transform_disease_data
    
    print("Running Full ETL Pipeline...")
    
    raw_fda = fetch_openfda_data(limit=500)
    raw_disease = fetch_disease_data()
    
    df_fda = transform_fda_data(raw_fda)
    df_disease = transform_disease_data(raw_disease)
    
    load_fda_to_supabase(df_fda)
    load_disease_to_supabase(df_disease)
    
    print("ETL Pipeline Complete!")