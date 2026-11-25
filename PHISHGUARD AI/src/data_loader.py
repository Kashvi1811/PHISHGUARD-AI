"""
PhishGuard AI - Data Loader Module
Dataset Acquisition from UCI Machine Learning Repository (Dataset ID: 327)
Official Name: Phishing Websites
"""

import os
import json
import logging
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

UCI_DATASET_ID = 327
UCI_OFFICIAL_URL = "https://archive.ics.uci.edu/dataset/327/phishing+websites"
UCI_ARFF_ZIP_URL = "https://archive.ics.uci.edu/static/public/327/phishing+websites.zip"


def download_uci_dataset(data_dir: str = "data/raw") -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Downloads UCI Phishing Websites dataset (ID 327) using ucimlrepo with robust fallback.
    Saves raw files in data_dir.
    
    Returns:
        X (pd.DataFrame): Raw features dataframe
        y (pd.DataFrame): Raw targets dataframe
        metadata (dict): Dataset metadata and citation info
    """
    os.makedirs(data_dir, exist_ok=True)
    raw_combined_path = os.path.join(data_dir, "phishing_raw.csv")
    raw_features_path = os.path.join(data_dir, "raw_features.csv")
    raw_targets_path = os.path.join(data_dir, "raw_targets.csv")
    metadata_path = os.path.join(data_dir, "metadata.json")

    X = None
    y = None
    metadata = {}

    # Method 1: Use official ucimlrepo package
    try:
        logger.info(f"Attempting to fetch dataset ID {UCI_DATASET_ID} via ucimlrepo...")
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=UCI_DATASET_ID)
        X = dataset.data.features
        y = dataset.data.targets
        
        # Extract metadata
        metadata = {
            "uci_id": dataset.metadata.get("uci_id", UCI_DATASET_ID),
            "name": dataset.metadata.get("name", "Phishing Websites"),
            "repository_url": dataset.metadata.get("repository_url", UCI_OFFICIAL_URL),
            "num_instances": int(dataset.metadata.get("num_instances", len(X))),
            "num_features": int(dataset.metadata.get("num_features", X.shape[1])),
            "target_col": dataset.metadata.get("target_col", ["result"]),
            "creators": dataset.metadata.get("creators", ["Rami Mohammad", "Lee McCluskey", "Fadi Thabtah"]),
            "year": dataset.metadata.get("year_of_dataset_creation", 2012),
            "doi": dataset.metadata.get("dataset_doi", "10.24432/C51W2X"),
            "citation": "Mohammad, R., Thabtah, F., & McCluskey, L. (2012). An assessment of features related to phishing websites using an automated technique. International Conference for Internet Technology and Secured Transactions."
        }
        logger.info("Successfully fetched dataset from UCI via ucimlrepo.")
    except Exception as e:
        logger.warning(f"ucimlrepo fetch failed: {e}. Attempting fallback download...")

    # Method 2: Fallback to downloading official UCI ARFF zip
    if X is None or y is None:
        try:
            import urllib.request
            import zipfile
            import io
            logger.info(f"Downloading official zip from {UCI_ARFF_ZIP_URL}...")
            req = urllib.request.Request(
                UCI_ARFF_ZIP_URL, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                zip_bytes = resp.read()
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                arff_names = [n for n in z.namelist() if n.endswith('.arff')]
                arff_name = 'Training Dataset.arff' if 'Training Dataset.arff' in arff_names else arff_names[0]
                content = z.read(arff_name).decode('utf-8', errors='ignore')
                
                # Parse ARFF
                attributes = []
                data_rows = []
                data_started = False
                for line in content.splitlines():
                    line = line.strip()
                    if not line or line.startswith('%'):
                        continue
                    if line.lower().startswith('@attribute'):
                        parts = line.split()
                        attr_name = parts[1].strip("'\"").lower()
                        attributes.append(attr_name)
                    elif line.lower().startswith('@data'):
                        data_started = True
                    elif data_started:
                        vals = [int(v.strip()) for v in line.split(',') if v.strip()]
                        if len(vals) == len(attributes):
                            data_rows.append(vals)
                
                df_all = pd.DataFrame(data_rows, columns=attributes)
                target_col = attributes[-1]
                X = df_all.iloc[:, :-1]
                y = df_all[[target_col]].rename(columns={target_col: 'result'})
                
                metadata = {
                    "uci_id": UCI_DATASET_ID,
                    "name": "Phishing Websites",
                    "repository_url": UCI_OFFICIAL_URL,
                    "num_instances": len(df_all),
                    "num_features": X.shape[1],
                    "target_col": ["result"],
                    "creators": ["Rami Mohammad", "Lee McCluskey", "Fadi Thabtah"],
                    "year": 2012,
                    "doi": "10.24432/C51W2X",
                    "citation": "Mohammad, R., Thabtah, F., & McCluskey, L. (2012). An assessment of features related to phishing websites using an automated technique."
                }
                logger.info("Successfully parsed dataset from official ARFF fallback.")
        except Exception as e2:
            logger.error(f"Fallback download also failed: {e2}")
            # Check if local files already exist
            if os.path.exists(raw_features_path) and os.path.exists(raw_targets_path):
                logger.info("Loading existing cached local files from data/raw/...")
                X = pd.read_csv(raw_features_path)
                y = pd.read_csv(raw_targets_path)
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            else:
                raise RuntimeError("Could not acquire UCI dataset via any method.")

    # Standardize column names (lowercase, underscores)
    X.columns = [col.lower().strip().replace(' ', '_').replace('-', '_') for col in X.columns]
    y.columns = ['result']

    # Save raw data locally
    raw_df = pd.concat([X, y], axis=1)
    raw_df.to_csv(raw_combined_path, index=False)
    X.to_csv(raw_features_path, index=False)
    y.to_csv(raw_targets_path, index=False)
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        
    logger.info(f"Raw dataset successfully saved to {data_dir}. Shape: X={X.shape}, y={y.shape}")
    return X, y, metadata


def generate_dataset_documentation(metadata: Dict[str, Any], doc_path: str = "data/dataset_info.md"):
    """
    Generates dataset_info.md detailing source attribution, feature descriptions, and label semantics.
    """
    doc_content = f"""# UCI Machine Learning Repository: Phishing Websites Dataset

## Overview & Metadata
- **Dataset Name:** {metadata.get('name', 'Phishing Websites')}
- **UCI ID:** {metadata.get('uci_id', 327)}
- **Official Repository URL:** [{metadata.get('repository_url', UCI_OFFICIAL_URL)}]({metadata.get('repository_url', UCI_OFFICIAL_URL)})
- **Dataset DOI:** `{metadata.get('doi', '10.24432/C51W2X')}`
- **Year of Creation:** {metadata.get('year', 2012)}
- **Creators:** {', '.join(metadata.get('creators', ['Rami Mohammad', 'Lee McCluskey', 'Fadi Thabtah']))}
- **Number of Instances:** {metadata.get('num_instances', 11055):,}
- **Number of Input Features:** {metadata.get('num_features', 30)}
- **Target Variable:** `result`

---

## Citation & Academic Attribution
```bibtex
@article{{mohammad2012assessment,
  title={{An assessment of features related to phishing websites using an automated technique}},
  author={{Mohammad, Rami M and Thabtah, Fadi and McCluskey, Lee}},
  journal={{International Conference for Internet Technology and Secured Transactions}},
  year={{2012}}
}}
```

---

## Target Label Semantics & Normalization

In the original UCI dataset, website classification is recorded in the `result` attribute as:
- **`1`**: Legitimate Website (6,157 instances, ~55.69%)
- **`-1`**: Phishing Website (4,898 instances, ~44.31%)

### Standardized Binary Normalization (Applied in PhishGuard AI):
To align with standard cybersecurity threat detection paradigms where **Class 1 represents the threat (anomaly/positive target)** and **Class 0 represents benign traffic**:

- **Class 0: Legitimate Website** (Benign, safe web portal) [Original Result = 1]
- **Class 1: Phishing Website** (Malicious attack vector) [Original Result = -1]

---

## Feature Taxonomy (30 Website Attributes)

The dataset contains 30 discrete attributes categorized across 4 threat dimensions:

### 1. Address Bar-Based Features
1. `having_ip_address`: Using IP instead of domain name (`-1`: Phishing, `1`: Legitimate)
2. `url_length`: Length of URL (`-1`: Phishing >= 54, `0`: Suspicious 54-75, `1`: Legitimate < 54)
3. `shortining_service`: Using URL shortener (e.g. bit.ly, tinyurl) (`-1`: Phishing, `1`: Legitimate)
4. `having_at_symbol`: '@' symbol ignores previous URL part (`-1`: Phishing, `1`: Legitimate)
5. `double_slash_redirecting`: '//' appearing after position 7 (`-1`: Phishing, `1`: Legitimate)
6. `prefix_suffix`: Dash '-' symbol added to domain name (`-1`: Phishing, `1`: Legitimate)
7. `having_sub_domain`: Number of subdomains (`-1`: Phishing > 2, `0`: Suspicious = 2, `1`: Legitimate = 1)
8. `sslfinal_state`: HTTPS & Certificate Trust/Age (`-1`: Phishing, `0`: Suspicious, `1`: Legitimate)
9. `domain_registration_length`: Domain expiry window (`-1`: Phishing <= 1 yr, `1`: Legitimate > 1 yr)
10. `favicon`: Favicon loaded from external/different domain (`-1`: Phishing, `1`: Legitimate)
11. `port`: Preferred non-standard open ports (`-1`: Phishing, `1`: Legitimate)
12. `https_token`: 'https' token embedded inside domain name (`-1`: Phishing, `1`: Legitimate)

### 2. Abnormal Features
13. `request_url`: Percentage of external objects loaded (`-1`: Phishing, `1`: Legitimate)
14. `url_of_anchor`: Percentage of anchors pointing to different/null domains (`-1`: Phishing, `0`: Suspicious, `1`: Legitimate)
15. `links_in_tags`: Ratio of links inside <meta>, <script>, <link> (`-1`: Phishing, `0`: Suspicious, `1`: Legitimate)
16. `sfh`: Server Form Handler pointing to blank or external URL (`-1`: Phishing, `0`: Suspicious, `1`: Legitimate)
17. `submitting_to_email`: Form action using mailto: (`-1`: Phishing, `1`: Legitimate)
18. `abnormal_url`: Hostname not matching whois identity (`-1`: Phishing, `1`: Legitimate)

### 3. HTML and JavaScript Based Features
19. `redirect`: Number of page redirects (`-1`: Phishing >= 4, `0`: Suspicious 2-3, `1`: Legitimate <= 1)
20. `on_mouseover`: Status bar manipulation (`-1`: Phishing, `1`: Legitimate)
21. `rightclick`: Context menu / right click disabled (`-1`: Phishing, `1`: Legitimate)
22. `popupwindow`: Pop-up window prompting for credentials (`-1`: Phishing, `1`: Legitimate)
23. `iframe`: Invisible iframe usage (`-1`: Phishing, `1`: Legitimate)

### 4. Domain & Reputation Features
24. `age_of_domain`: Whois domain age >= 6 months (`-1`: Phishing, `1`: Legitimate)
25. `dnsrecord`: DNS record found in authority server (`-1`: Phishing, `1`: Legitimate)
26. `web_traffic`: Alexa / SimilarWeb rank (`-1`: Phishing, `0`: Suspicious, `1`: Legitimate)
27. `page_rank`: Google PageRank score (`-1`: Phishing, `1`: Legitimate)
28. `google_index`: Webpage indexed by Google Search (`-1`: Phishing, `1`: Legitimate)
29. `links_pointing_to_page`: External inbound link count (`-1`: Phishing, `0`: Suspicious, `1`: Legitimate)
30. `statistical_report`: Host found in PhishTank/StopBadware report (`-1`: Phishing, `1`: Legitimate)
"""
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(doc_content)
    logger.info(f"Dataset documentation saved to {doc_path}")


if __name__ == "__main__":
    X_raw, y_raw, meta = download_uci_dataset()
    generate_dataset_documentation(meta)
    print(f"Data loading complete! X shape: {X_raw.shape}, y value counts:\n{y_raw.value_counts()}")
