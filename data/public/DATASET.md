# Public dataset used in the demo

**File:** `data/public/phishing_urls_10k.csv` (10,000 rows, 18 columns, 5,000 phishing and 5,000 legitimate)

**Source:** Shreya Gopal, "Phishing Website Detection by Machine Learning Techniques", GitHub repository
https://github.com/shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques (file `DataFiles/5.urldata.csv`).

**Where the rows come from (as described by that repository):**
- Phishing URLs: 5,000 random URLs from PhishTank (https://www.phishtank.com/developer_info.php).
- Legitimate URLs: 5,000 random benign URLs from the University of New Brunswick URL 2016 dataset (https://www.unb.ca/cic/datasets/url-2016.html).
- Features: 17 address-bar, domain and HTML/JavaScript features extracted from each URL, following the feature definitions of the UCI "Phishing Websites" dataset (https://archive.ics.uci.edu/ml/datasets/Phishing+Websites).

**Columns:** `Domain` (text), 16 binary features (`Have_IP`, `Have_At`, `URL_Length`, `URL_Depth`, `Redirection`, `https_Domain`, `TinyURL`, `Prefix/Suffix`, `DNS_Record`, `Web_Traffic`, `Domain_Age`, `Domain_End`, `iFrame`, `Mouse_Over`, `Right_Click`, `Web_Forwards`) and `Label` (1 = phishing, 0 = legitimate).

**Notes:** the file contains many repeated rows (the app's cleaner removes duplicates). Use it for education only, and check the source repository for its terms before reusing it elsewhere.
