import requests

def fetch_openfda_data(limit=100):
    """OpenFDA API se latest drug adverse events ka data extract karna"""
    url = f"https://api.fda.gov/drug/event.json?limit={limit}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            print("Successfully fetched OpenFDA data!")
            return response.json().get('results', [])
        else:
            print(f"OpenFDA Error: Status Code {response.status_code}")
            return []
    except Exception as e:
        print(f"Exception fetching OpenFDA: {e}")
        return []

def fetch_disease_data():
    url = "https://disease.sh/v3/covid-19/historical/all?lastdays=100"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            print("Successfully fetched Disease.sh Health data!")
            return response.json()
        else:
            print(f"Disease.sh Error: Status Code {response.status_code}")
            return {}
    except Exception as e:
        print(f"Exception fetching Disease.sh: {e}")
        return {}

if __name__ == "__main__":
    print("Testing Ingestion Layer with Open Platform...")
    fda_sample = fetch_openfda_data(limit=1000)
    disease_sample = fetch_disease_data()
    
    fda_count = len(fda_sample)
    disease_count = len(disease_sample.get('cases', {})) if disease_sample else 0
    
    print(f"\n--- Final Result ---")
    print(f"Fetched {fda_count} OpenFDA records.")
    print(f"Fetched {disease_count} days of Historical Disease trends.")