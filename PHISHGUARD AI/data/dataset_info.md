# UCI Machine Learning Repository: Phishing Websites Dataset

## Overview & Metadata
- **Dataset Name:** Phishing Websites
- **UCI ID:** 327
- **Official Repository URL:** [https://archive.ics.uci.edu/dataset/327/phishing+websites](https://archive.ics.uci.edu/dataset/327/phishing+websites)
- **Dataset DOI:** `10.24432/C51W2X`
- **Year of Creation:** 2012
- **Creators:** Rami Mohammad, Lee McCluskey
- **Number of Instances:** 11,055
- **Number of Input Features:** 30
- **Target Variable:** `result`

---

## Citation & Academic Attribution
```bibtex
@article{mohammad2012assessment,
  title={An assessment of features related to phishing websites using an automated technique},
  author={Mohammad, Rami M and Thabtah, Fadi and McCluskey, Lee},
  journal={International Conference for Internet Technology and Secured Transactions},
  year={2012}
}
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
