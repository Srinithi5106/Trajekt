import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

OUTPUT_DIR = "dataset_synthetic"

def generate_synthetic_data():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    rows = []
    months_count = 6
    start_date = datetime(2001, 1, 15)

    background_users = [f"peer{i}@enron.com" for i in range(10)]
    
    for m in range(months_count):
        month_dt = start_date + timedelta(days=m*30)
        
        # 1. user_stable@enron.com
        for e in range(20):
            rows.append({
                "sender": "user_stable@enron.com",
                "recipient": background_users[e % len(background_users)],
                "timestamp": (month_dt + timedelta(hours=e)).isoformat() + "Z",
                "department": "DCAR"
            })
            
        # 2. user_promoted@enron.com
        dept = "DISQ" if m < 3 else "DMCT"
        for e in range(20):
            rows.append({
                "sender": "user_promoted@enron.com",
                "recipient": background_users[(e+1) % len(background_users)],
                "timestamp": (month_dt + timedelta(hours=e)).isoformat() + "Z",
                "department": dept
            })
            
        # 3. user_resigned@enron.com
        if m < 3: vol = 100
        elif m == 3: vol = 80
        elif m == 4: vol = 60
        else: vol = 45
        for e in range(vol):
            rows.append({
                "sender": "user_resigned@enron.com",
                "recipient": background_users[e % len(background_users)],
                "timestamp": (month_dt + timedelta(minutes=e*15)).isoformat() + "Z",
                "department": "DMI"
            })
            
        # 4. user_fired@enron.com
        vol = 50 if m < 5 else 2
        for e in range(vol):
            rows.append({
                "sender": "user_fired@enron.com",
                "recipient": background_users[(e+2) % len(background_users)],
                "timestamp": (month_dt + timedelta(hours=e)).isoformat() + "Z",
                "department": "DSE"
            })

        # 5. user_bottleneck@enron.com
        for peer in background_users:
            for _ in range(5):
                rows.append({
                    "sender": peer,
                    "recipient": "user_bottleneck@enron.com",
                    "timestamp": (month_dt + timedelta(minutes=np.random.randint(1, 60))).isoformat() + "Z",
                    "department": "DCAR"
                })
        out_vol = 10 if m < 5 else 1
        for e in range(out_vol):
            rows.append({
                "sender": "user_bottleneck@enron.com",
                "recipient": background_users[e],
                "timestamp": (month_dt + timedelta(hours=e+1)).isoformat() + "Z",
                "department": "DST"
            })

        # 6. user_isolated@enron.com
        rows.append({
            "sender": "user_isolated@enron.com",
            "recipient": "peer0@enron.com",
            "timestamp": (month_dt + timedelta(hours=1)).isoformat() + "Z",
            "department": "SCOM"
        })
        rows.append({
            "sender": "user_isolated@enron.com",
            "recipient": "peer1@enron.com",
            "timestamp": (month_dt + timedelta(hours=2)).isoformat() + "Z",
            "department": "SCOM"
        })

    df_synthetic = pd.DataFrame(rows)
    df_synthetic = df_synthetic.sample(frac=1, random_state=42).reset_index(drop=True)
    
    out_path = os.path.join(OUTPUT_DIR, "email_edges_synthetic.csv")
    df_synthetic.to_csv(out_path, index=False)
    
    print(f"Generated {len(df_synthetic)} synthetic email edges.")
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    generate_synthetic_data()
