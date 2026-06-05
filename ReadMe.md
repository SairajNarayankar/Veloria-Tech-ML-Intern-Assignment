<div align="center">

# 🏏 Cricket AI/ML Predictive Analytics & RAG Search Suite

<img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/BeautifulSoup4-59666C?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/ChromaDB-FF6B6B?style=for-the-badge&logo=databricks&logoColor=white"/>
<img src="https://img.shields.io/badge/Status-Production%20Ready-00C851?style=for-the-badge"/>

<br/>

> **AI/ML Internship Assignment** — A production-grade, end-to-end data engineering and machine learning pipeline for cricket match analytics.

<br/>

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   scraper.py  →  match_data.csv  →  model.py           │
│       │                                 │               │
│   [Web Scrape]              [ML Predict + Evaluate]    │
│                                         │               │
│                            rag_search.py ←              │
│                         [Semantic RAG Search]           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

</div>

---

## 📌 Project Overview

This project delivers a **complete, interconnected data science pipeline** across three tasks:

| # | Script | Purpose | Output |
|---|--------|---------|--------|
| 1 | `scraper.py` | Web scraping cricket match data | `match_data.csv` |
| 2 | `model.py` | ML match outcome prediction | Accuracy: **80%**, F1: **0.8571** |
| 3 | `rag_search.py` | Semantic search with RAG pipeline | Natural language match search |

**Teams analysed:** 🇮🇳 India &nbsp;·&nbsp; 🇦🇺 Australia &nbsp;&nbsp;|&nbsp;&nbsp; **Matches per team:** 10 &nbsp;&nbsp;|&nbsp;&nbsp; **Format:** ODI

---

## ⚙️ Installation

```bash
pip install requests beautifulsoup4 pandas scikit-learn numpy chromadb sentence-transformers
```

> Requires **Python 3.8+**

---

## 📂 Task 1 — Web Scraping `scraper.py`

### What It Does

Crawls **HowStat's** historical ODI archives for the last 10 completed matches per team, follows each scorecard link, and extracts the following fields:

| Field | Example |
|-------|---------|
| Match Date | `22 Nov 2024` |
| Team 1 / Team 2 | `Australia` / `India` |
| Venue | `Optus Stadium, Perth` |
| Result | `India won by 295 runs` |
| Top Scorer | `Yashasvi Jaiswal (161)` |

### How It Works

```
Step 1 │ POST request to HowStat's MatchListCountry_ODI.asp
       │ with country codes (IND / AUS)
       ↓
Step 2 │ Backward Verification Loop
       │ Starts from bottom of table (most recent matches)
       │ Skips: "abandoned" / "no result" / "cancelled"
       ↓
Step 3 │ Deep Scorecard Crawling
       │ Follows individual match URLs
       │ Sorts all batsmen by runs → extracts top scorer
       ↓
Step 4 │ Save to match_data.csv
```

### Why HowStat Over ESPN Cricinfo?

> Modern sites like ESPN Cricinfo use **Akamai/Cloudflare anti-bot CDNs** that throw `403 Forbidden` errors even with advanced headers. HowStat guarantees **100% scraper stability** without headless browser automation.

**Resiliency Features:**
- ⏱️ `time.sleep(0.5)` polite rate limiting between requests
- 🔁 3-stage retry-on-failure block per scorecard fetch
- 🧹 Robust try/except wrappers throughout

### Run

```bash
python scraper.py
```

**Output:** Console progress logs + `match_data.csv` in working directory
<img width="524" height="532" alt="image" src="https://github.com/user-attachments/assets/3eb279aa-2111-4b21-9e5c-998d02073a42" />


---

## 🤖 Task 2 — ML Prediction Model `model.py`

### What It Does

Trains a **Random Forest Classifier** on `match_data.csv` to predict which team wins a given match, using carefully engineered features that prevent data leakage.

### Feature Engineering

#### 1. Chronological Rolling Team Form — `team_win_rate`

```
❌ Naive approach: Calculate overall win-rate (introduces data leakage)

✅ Our approach:
   Sort all matches chronologically
   For each match → calculate win/loss record using ONLY prior matches
   No future information ever touches a past prediction
```

#### 2. Home-Ground Advantage

```python
is_team1_home   # 1 if team plays in their own country
is_team2_home   # 1 if team plays in their own country  
is_neutral      # 1 if venue is in a third country (e.g. UAE)
```

Venue cities (e.g. *Ranchi*, *Sydney*) are mapped to countries and encoded as indicator flags.

#### 3. Label Encoding

Team names and venues are converted to numeric arrays using Scikit-Learn's `LabelEncoder` for model compatibility.

### Algorithm Selection

| Model | Accuracy | Notes |
|-------|----------|-------|
| Logistic Regression | **40.0%** | Fails to capture non-linear interactions on small data |
| **Random Forest** ✅ | **80.0%** | Ensemble of trees, handles non-linearity, resists overfitting |

Random Forest averages across multiple decision trees — ideal for the non-linear interaction of *rolling form × away-game pressure × team-specific tendencies*.

### Performance Metrics

**Test Split: 25% Stratified**

```
┌─────────────────────────────────────────┐
│  Accuracy Score  :  0.8000   (80.0%)   │
│  F1 Score        :  0.8571              │
│                                         │
│  Confusion Matrix:                      │
│    [[1, 1],  ← True Neg, False Pos     │
│     [0, 3]]  ← False Neg, True Pos     │
│                                         │
│  Only 1 error on the entire test set!  │
└─────────────────────────────────────────┘
```

### Run

```bash
python model.py
```

**Output:** Step-by-step progress + detailed evaluation report with real vs. predicted comparison
<img width="589" height="367" alt="image" src="https://github.com/user-attachments/assets/ac9bc705-3405-4f57-98f9-cf2a167b927e" />


---

## 🔍 Task 3 — Semantic RAG Search `rag_search.py`

### What It Does

Applies **Retrieval-Augmented Generation (RAG)** concepts to enable natural language search over the match dataset. Instead of exact keyword matching, ask conceptual questions:

```
"Show me matches where the away team won"
"Who scored the highest runs?"
"Recent games played in UAE"
```

The engine understands context and retrieves the **top 3 most relevant matches**.

### How It Works

```
Step 1 │ TEXT SERIALIZATION
       │ Structured row → Rich natural prose
       │
       │  Row:   18/01/2026, India, New Zealand, Indore, NZ won, D.Mitchell, 137
       │  Prose: "India vs New Zealand at Holkar Cricket Stadium, Indore on
       │          18/01/2026. Result: New Zealand won by 41 runs.
       │          Top Scorer: Daryl Mitchell with 137 runs."
       ↓
Step 2 │ NEURAL VECTOR EMBEDDINGS
       │ all-MiniLM-L6-v2 SentenceTransformer
       │ → 384-dimensional dense vectors per match
       ↓
Step 3 │ VECTOR DATABASE STORAGE
       │ Local ChromaDB collection
       ↓
Step 4 │ SEMANTIC SEARCH
       │ Cosine similarity: query vector ↔ match vectors
       │ → Top 3 results returned
```

### Dual-Engine Resiliency Architecture

```
                     ┌─────────────────────┐
    Query Input ───► │   Primary Engine    │
                     │  SentenceTransfor.  │
                     │  + ChromaDB         │
                     └────────┬────────────┘
                              │  If import fails
                              ▼
                     ┌─────────────────────┐
                     │   Fallback Engine   │
                     │  Scikit-Learn       │
                     │  TF-IDF + Cosine    │
                     └─────────────────────┘
```

> **Why this matters:** `torch` and `sentence-transformers` are heavy. On machines with limited disk/memory, installs can fail. The fallback guarantees **100% executable portability** — the evaluator runs the script instantly on any system.

### Run

```bash
python rag_search.py
```
<img width="1612" height="667" alt="image" src="https://github.com/user-attachments/assets/bd0b956d-bb82-47fc-bb9f-19bcf92e5f19" />


**Interactive Mode** — Live terminal prompt:
```
Search Query > Show me matches where India won away
```

**Automated Mode** — If run in a non-interactive console (e.g. grading pipelines), executes a preset series of test queries automatically.

---

## 🌟 Key Design Decisions & Best Practices

| Practice | Implementation |
|----------|---------------|
| 🔒 **Zero Data Leakage** | Rolling form calculated chronologically — model never sees future data |
| 🤝 **Polite Scraping** | Adaptive timeouts, browser-like headers, built-in rate limiting |
| 🛡️ **Fail-Safe Architecture** | Dual-engine RAG search — never crashes due to missing libraries |
| 🔑 **No External Keys** | Fully self-contained — no cloud DB, no API keys required |
| ♻️ **Modular Pipeline** | Each script is independently runnable yet feeds into the next |

---

## 📁 Repository Structure

```
📦 veloria-tech-ml-intern-assignment/
 ┣ 📄 scraper.py          ← Task 1: Web scraper
 ┣ 📄 model.py            ← Task 2: ML prediction model
 ┣ 📄 rag_search.py       ← Task 3: Semantic RAG search
 ┣ 📄 match_data.csv      ← Generated dataset (20 matches)
 ┗ 📄 README.md           ← This file
```

---

<div align="center">

Made with 🏏 for the **Veloria Tech AI/ML Internship Assignment**

</div>
