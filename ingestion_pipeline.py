import os
import email
import pandas as pd
from email.policy import default
from email.utils import parseaddr
from dateutil import parser
from tqdm import tqdm
import re

# --- CONFIGURATION ---
ENRON_DIR = 'maildir'
SOCIOPATTERNS_PROXIMITY_FILE = 'tij_InVS13.dat'
SOCIOPATTERNS_METADATA_FILE = 'metadata_InVS13.txt'

OUTPUT_EMAIL_EDGES = 'email_edges.csv'
OUTPUT_AGGREGATED_EDGES = 'email_edges_aggregated.csv'
OUTPUT_PROXIMITY_EDGES = 'proximity_edges.csv'
OUTPUT_NODE_DEPARTMENTS = 'node_departments.csv'
OUTPUT_SAMPLED_EMAIL_EDGES = 'email_edges_sampled.csv'

# Performance: Set to an integer (e.g., 5000) for fast testing, or None for production
LIMIT_FILES = None

# Filtering logic
START_YEAR = 1999
END_YEAR = 2002

# Enhanced Department Mapping
DEPT_MAPPING = {
    "legal": "legal",
    "trading": "trading",
    "finance": "finance",
    "accounting": "finance",
    "hr": "hr",
    "human resources": "hr",
    "marketing": "marketing",
    "exec": "executive",
    "operations": "operations"
}

def clean_email(email_str):
    """Extracts pure email address, lowercases, and strips whitespace."""
    if not email_str:
        return None
    # Use standard email utility to parse "Name <email@domain.com>"
    _, address = parseaddr(email_str)
    address = address.lower().strip()
    # Ensure it looks like a valid email via basic regex
    if re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', address):
        return address
    return None

def infer_department(file_path):
    """Infers department from folder path keywords."""
    path_lower = file_path.lower().replace('\\', '/')
    for keyword, dept in DEPT_MAPPING.items():
        if keyword in path_lower:
            return dept
    return "unknown"

def parse_enron_emails(maildir_path, limit=None):
    """Parses Enron maildir with normalization, time-filtering, and robust error handling."""
    print(f"🚀 [PRODUCTION] Starting Enron Email Parsing: {maildir_path}")
    
    if not os.path.exists(maildir_path):
        print(f"❌ Error: {maildir_path} not found.")
        return pd.DataFrame(), 0, 0

    email_data = []
    skipped_count = 0
    total_processed = 0
    
    all_files = []
    for root, dirs, files in os.walk(maildir_path):
        for file in files:
            all_files.append(os.path.join(root, file))
            if limit and len(all_files) >= limit:
                break
        if limit and len(all_files) >= limit:
            break
    
    with tqdm(total=len(all_files), desc="Processing Emails") as pbar:
        for file_path in all_files:
            total_processed += 1
            dept = infer_department(file_path)
            
            # Robust file path handling for Windows (trailing dots issue)
            if os.name == 'nt':
                abs_path = os.path.join(os.getcwd(), file_path)
                if not abs_path.startswith('\\\\?\\'):
                    actual_path = '\\\\?\\' + abs_path
                else:
                    actual_path = abs_path
            else:
                actual_path = file_path

            try:
                with open(actual_path, 'r', encoding='utf-8', errors='ignore') as f:
                    msg = email.message_from_file(f, policy=default)
                    
                    sender_raw = msg.get('From', '')
                    sender = clean_email(sender_raw)
                    to_header = msg.get('To', '')
                    date_str = msg.get('Date', '')
                    
                    if not sender or not to_header or not date_str:
                        skipped_count += 1
                        pbar.update(1)
                        continue
                        
                    # Time Filtering
                    try:
                        dt = parser.parse(date_str)
                        if not (START_YEAR <= dt.year <= END_YEAR):
                            skipped_count += 1
                            pbar.update(1)
                            continue
                        timestamp = dt.isoformat()
                    except Exception:
                        skipped_count += 1
                        pbar.update(1)
                        continue
                        
                    # Recipient Normalization
                    raw_recipients = to_header.split(',')
                    found_recipient = False
                    for r_raw in raw_recipients:
                        recipient = clean_email(r_raw)
                        if recipient:
                            # Self-loop check
                            if sender != recipient:
                                email_data.append({
                                    'sender': sender,
                                    'recipient': recipient,
                                    'timestamp': timestamp,
                                    'department': dept
                                })
                                found_recipient = True
                    
                    if not found_recipient:
                        skipped_count += 1
                    else:
                        pass # Successfully added at least one recipient
            except Exception:
                skipped_count += 1
            
            pbar.update(1)

    df = pd.DataFrame(email_data)
    return df, total_processed, skipped_count

def process_sociopatterns(proximity_file, metadata_file):
    """Processes SocioPatterns proximity and metadata files."""
    print("🚀 Processing SocioPatterns data...")
    prox_df = pd.DataFrame()
    meta_df = pd.DataFrame()
    
    if os.path.exists(proximity_file):
        prox_df = pd.read_csv(proximity_file, sep=' ', header=None, names=['timestamp', 'i', 'j'])
        prox_df['duration'] = 20
        prox_df.to_csv(OUTPUT_PROXIMITY_EDGES, index=False)
        print(f"✅ Saved {OUTPUT_PROXIMITY_EDGES}")
    
    if os.path.exists(metadata_file):
        meta_df = pd.read_csv(metadata_file, sep='\t', header=None, names=['node_id', 'department'])
        meta_df.to_csv(OUTPUT_NODE_DEPARTMENTS, index=False)
        print(f"✅ Saved {OUTPUT_NODE_DEPARTMENTS}")
        
    return prox_df, meta_df

def main():
    # 1. Parsing and Normalization
    email_df, total_proc, skipped = parse_enron_emails(ENRON_DIR, limit=LIMIT_FILES)
    
    if email_df.empty:
        print("⚠️ No valid email data extracted.")
        # Attempt to process SocioPatterns even if Enron fails
        process_sociopatterns(SOCIOPATTERNS_PROXIMITY_FILE, SOCIOPATTERNS_METADATA_FILE)
    else:
        # 2. Deduplication & Sorting
        initial_len = len(email_df)
        email_df.drop_duplicates(inplace=True)
        email_df.sort_values('timestamp', inplace=True)
        
        # Save primary edges
        email_df.to_csv(OUTPUT_EMAIL_EDGES, index=False)
        print(f"✅ Saved {OUTPUT_EMAIL_EDGES}")

        # 3. Bonus: Aggregated Edges (Weighted)
        agg_df = email_df.groupby(['sender', 'recipient']).size().reset_index(name='weight')
        agg_df.to_csv(OUTPUT_AGGREGATED_EDGES, index=False)
        print(f"✅ Saved {OUTPUT_AGGREGATED_EDGES}")

        # 4. Sampling
        activity = pd.concat([email_df['sender'], email_df['recipient']]).value_counts()
        top_200_users = activity.head(200).index.tolist()
        sampled_df = email_df[email_df['sender'].isin(top_200_users) & email_df['recipient'].isin(top_200_users)]
        sampled_df.to_csv(OUTPUT_SAMPLED_EMAIL_EDGES, index=False)
        print(f"✅ Saved {OUTPUT_SAMPLED_EMAIL_EDGES}")
        
        # 5. SocioPatterns
        prox_df, meta_df = process_sociopatterns(SOCIOPATTERNS_PROXIMITY_FILE, SOCIOPATTERNS_METADATA_FILE)

        # 6. Production Validation Report
        nodes = set(email_df['sender']).union(set(email_df['recipient']))
        unknown_dept_pct = (email_df['department'] == 'unknown').mean() * 100
        
        print("\n" + "═"*45)
        print("📊 PRODUCTION DATA VALIDATION REPORT")
        print("═"*45)
        print(f"Total files scanned:       {total_proc}")
        print(f"Valid edges extracted:     {len(email_df)}")
        print(f"Skipped/Time-filtered:     {skipped}")
        print(f"Duplicates removed:        {initial_len - len(email_df)}")
        print(f"Unique nodes:              {len(nodes)}")
        print(f"Unique departments:        {email_df['department'].nunique()}")
        print(f"Unknown department pct:    {unknown_dept_pct:.2f}%")
        print(f"Time Range:                {email_df['timestamp'].min()} → {email_df['timestamp'].max()}")
        print("═"*45)
        print("\n🏆 TOP 10 MOST ACTIVE USERS:")
        print(activity.head(10))
        print("═"*45)

if __name__ == "__main__":
    main()
