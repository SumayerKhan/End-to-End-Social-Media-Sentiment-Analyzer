# End-to-End Social Media Sentiment Analyzer

> Automated system for collecting, analyzing, and visualizing public sentiment about consumer electronics from Reddit discussions

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Database](https://img.shields.io/badge/Database-SQLite-blue.svg)]()
[![Status](https://img.shields.io/badge/Status-Week%203%20Complete-green.svg)]()

---

## 🎯 Project Overview

A complete pipeline for sentiment analysis of consumer electronics discussions on Reddit, evolving into an intelligent RAG-based Q&A system.

**System Components:**
- 🤖 **Automated Collector** - GitHub Actions collects data every 3 hours ✅
- 🗄️ **Database** - SQLite storage with sentiment scores ✅
- 🧠 **Sentiment Analysis** - VADER classification system ✅
- 🔄 **Automated Pipeline** - One-command data processing ✅
- 🔍 **API** - REST endpoints with FastAPI *(Coming Week 4)*
- 📊 **Dashboard** - Interactive Streamlit visualization *(Coming Week 5-6)*
- 🤖 **RAG Chatbot** - Natural language Q&A system *(Coming Week 7)*

**Current Progress:** ✅ Week 3 Complete | 17,479+ posts analyzed with sentiment scores

**Project Goal:** Complete RAG-based Q&A system for natural language queries about tech products (Week 7)

---

## 📈 Current Dataset Stats

- **Total Posts:** 17,479+ (and growing)
- **Subreddits Monitored:** 20 consumer electronics communities
- **Sentiment Analysis:** Complete (VADER-based classification)
- **Sentiment Distribution:**
  - Positive: 48.4%
  - Negative: 20.1%
  - Neutral: 31.5%
- **Collection Method:** Official Reddit API (PRAW) - ethical and compliant
- **Update Frequency:** Automated collection every 3 hours via GitHub Actions

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Reddit account (for API credentials)
- Git installed
- 100MB free disk space

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/SumayerKhan/End-to-End-Social-Media-Sentiment-Analyzer.git
cd End-to-End-Social-Media-Sentiment-Analyzer
```

2. **Create virtual environment**
```bash
python -m venv .venv

# Activate:
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure Reddit API**
   - Go to https://www.reddit.com/prefs/apps
   - Create a new app (script type)
   - Copy the credentials

5. **Set up environment variables**
```bash
# Copy template
cp .env.example .env

# Edit .env with your credentials:
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=sentiment_analyzer/1.0
```

---

## 🔄 Running the System

### Update Database with Latest Data (Recommended)

**Single command to update everything:**
```bash
python scripts/auto_pipeline.py
```

This automated pipeline:
1. Pulls latest JSON files from GitHub
2. Imports new posts to database
3. Runs VADER sentiment analysis on new posts
4. Displays results summary

**Output:**
- Shows number of new posts imported
- Displays sentiment analysis progress
- Provides sentiment breakdown statistics

---

### Individual Commands

**Check database statistics:**
```bash
python database/check_db.py
```

**View sentiment analysis results:**
```bash
python analyzer/show_results.py
```

**Preview random sample of posts:**
```bash
python database/preview_data.py
```

**Import JSON files manually:**
```bash
python scripts/import_from_github.py
```

**Run sentiment analysis manually:**
```bash
python analyzer/process_posts.py
```

---

## 📁 Project Structure
```
sentiment-analyzer/
├── collector/                    # Data collection (Weeks 1-2) ✅
│   ├── github_collector.py      # GitHub Actions collector
│   ├── continuous_collector.py  # Alternative local collector
│   ├── reddit_config.py         # Reddit API configuration
│   └── scheduler.py             # Collection scheduler
│
├── database/                     # Data storage ✅
│   ├── tech_sentiment.db        # SQLite database (local only)
│   ├── check_db.py             # Database statistics viewer
│   └── preview_data.py         # Data quality checker
│
├── analyzer/                     # Sentiment analysis (Week 3) ✅
│   ├── process_posts.py        # VADER sentiment processor
│   ├── show_results.py         # Results visualization
│   ├── add_sentiment_columns.py # Database schema updates
│   └── sentiment_analyzer.py   # VADER testing utilities
│
├── scripts/                      # Automation scripts ✅
│   ├── auto_pipeline.py        # Complete pipeline orchestrator
│   └── import_from_github.py   # JSON to SQLite importer
│
├── data/collected/               # JSON files from GitHub Actions
│   └── reddit_posts_*.json     # Timestamped collections (45+ files)
│
├── .github/workflows/            # GitHub Actions automation ✅
│   └── collect.yml             # Runs every 3 hours
│
├── api/                          # REST API (Week 4)
├── dashboard/                    # Streamlit UI (Week 5)
├── preprocessing/                # Text cleaning utilities
├── tests/                        # Unit tests
│
├── .env                          # Secrets (NOT in repo)
├── .env.example                  # Environment template
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── LICENSE                       # MIT License
```

---

## 🛠️ Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.13 | Core development |
| **Reddit API** | PRAW 7.8.1 | Data collection |
| **Database** | SQLite3 | Local data storage |
| **Sentiment Analysis** | VADER | Text sentiment classification ✅ |
| **Automation** | GitHub Actions | Scheduled data collection ✅ |
| **API Framework** | FastAPI | REST API endpoints *(Week 4)* |
| **Dashboard** | Streamlit | Interactive visualization *(Week 5)* |
| **Visualization** | Plotly/Recharts | Charts and graphs *(Week 5)* |

---

## 🎓 Key Features

### ✅ Implemented (Weeks 1-3)

**Data Collection:**
- [x] Multi-subreddit data collection (20 communities)
- [x] Three feed types per subreddit (new, hot, rising)
- [x] Quality filtering and spam detection
- [x] Automatic deduplication (INSERT OR IGNORE)
- [x] GitHub Actions automation (runs every 3 hours)

**Database:**
- [x] SQLite database with optimized indexes
- [x] 17,479+ posts stored
- [x] Sentiment score columns added
- [x] Data verification tools

**Sentiment Analysis:**
- [x] VADER integration and testing
- [x] Automated sentiment processing
- [x] Classification: positive/negative/neutral
- [x] Compound sentiment scores (-1 to +1)
- [x] Results visualization and statistics

**Automation:**
- [x] Complete data pipeline (import → analyze → report)
- [x] One-command execution
- [x] Modular, maintainable code structure
- [x] Error handling and logging

### 🔄 In Progress (Week 4)
- [ ] REST API with FastAPI
- [ ] Keyword search endpoints
- [ ] Sentiment query endpoints
- [ ] API documentation with OpenAPI

### 📅 Planned (Weeks 5-7)
- [ ] Interactive Streamlit dashboard
- [ ] Time-series sentiment visualization
- [ ] Filtering by subreddit, date, sentiment
- [ ] Top posts by sentiment scores
- [ ] Testing and deployment

---

## 🧠 Sentiment Analysis Implementation

### VADER (Valence Aware Dictionary and sEntiment Reasoner)

**Why VADER?**
- Specifically designed for social media text
- Handles slang, emojis, and internet language
- Understands emphasis (!!!, CAPS, emoticons)
- No training required
- Fast processing (~5,000 posts/minute)

**Classification Method:**
```python
# Compound score ranges:
if compound >= 0.05:  → Positive
elif compound <= -0.05: → Negative
else: → Neutral
```

**Database Schema:**
```sql
-- Sentiment columns added to raw_posts table
ALTER TABLE raw_posts ADD COLUMN sentiment_pos REAL;      -- Positive score (0-1)
ALTER TABLE raw_posts ADD COLUMN sentiment_neg REAL;      -- Negative score (0-1)
ALTER TABLE raw_posts ADD COLUMN sentiment_neu REAL;      -- Neutral score (0-1)
ALTER TABLE raw_posts ADD COLUMN sentiment_compound REAL; -- Overall score (-1 to +1)
ALTER TABLE raw_posts ADD COLUMN sentiment_label TEXT;    -- Category label
```

**Processing Pipeline:**
1. Fetch posts without sentiment scores
2. Combine title + selftext for analysis
3. Calculate VADER scores
4. Classify into positive/negative/neutral
5. Update database with results
6. Generate statistics and reports

---

## 🔄 Automated Pipeline

### System Architecture

```
GitHub Actions (Cloud - Every 3 hours)
    ↓
Collect Reddit posts → Save as JSON
    ↓
Commit to GitHub repository
    ↓
[YOUR COMPUTER - Run when needed]
    ↓
python scripts/auto_pipeline.py
    ↓
    ├─→ Step 1: Pull latest JSON files (git pull)
    ├─→ Step 2: Import JSON to SQLite (skip duplicates)
    ├─→ Step 3: Run VADER on new posts
    └─→ Step 4: Display results summary
    ↓
Updated local database with sentiment scores
```

### Why This Architecture?

**Advantages:**
- ✅ Automated data collection (no manual intervention)
- ✅ Version-controlled JSON files
- ✅ Local database (fast, no cloud costs)
- ✅ One command to update everything
- ✅ Modular design (easy to maintain)
- ✅ Cloud-ready (easy to deploy later)

**Files Involved:**
- `collector/github_collector.py` - Runs in GitHub Actions
- `scripts/import_from_github.py` - Imports JSON to database
- `analyzer/process_posts.py` - Runs sentiment analysis
- `scripts/auto_pipeline.py` - Orchestrates everything

---

## 📊 Development Timeline

**Week 1-2: Data Collection** ✅
- Implemented Reddit API integration
- Built continuous collector with quality filters
- Created database schema with indexes
- Set up GitHub Actions automation
- Collected 17,479+ posts from 20 subreddits

**Week 3: Sentiment Analysis** ✅
- Integrated VADER sentiment analyzer
- Added sentiment columns to database
- Processed all posts with sentiment scores
- Created automated pipeline script
- Built results visualization tools
- Completed documentation

**Week 4: REST API** 🔄
- Build REST API with FastAPI
- Create query endpoints
- Implement filtering and search
- Generate API documentation

**Week 5-6: Dashboard** 📅
- Develop Streamlit dashboard
- Create interactive sentiment charts
- Add filtering and search UI
- Implement data export features

**Week 7: Testing & Polish** 📅
- End-to-end testing
- Performance optimization
- Final presentation preparation

---

## 📈 Current Results

### Sentiment Distribution (17,479 posts)

**Overall Breakdown:**
- **Positive:** 8,456 posts (48.4%)
- **Negative:** 3,512 posts (20.1%)
- **Neutral:** 5,511 posts (31.5%)

**Top Communities by Volume:**
1. r/pcmasterrace - 3,031 posts
2. r/buildapc - 2,951 posts
3. r/TechSupport - 2,565 posts
4. r/iphone - 1,708 posts
5. r/laptops - 1,276 posts

**Analysis Period:** October 19-26, 2025

---

## 🗂️ Database Schema

### `raw_posts` Table
```sql
CREATE TABLE raw_posts (
    -- Original columns
    post_id TEXT PRIMARY KEY,
    subreddit TEXT NOT NULL,
    title TEXT NOT NULL,
    selftext TEXT,
    author TEXT,
    created_utc REAL NOT NULL,
    score INTEGER,
    num_comments INTEGER,
    url TEXT,
    permalink TEXT,
    collected_at REAL NOT NULL,
    
    -- Sentiment columns (Week 3)
    sentiment_pos REAL,
    sentiment_neg REAL,
    sentiment_neu REAL,
    sentiment_compound REAL,
    sentiment_label TEXT
);

-- Indexes for performance
CREATE INDEX idx_title ON raw_posts(title);
CREATE INDEX idx_created ON raw_posts(created_utc DESC);
CREATE INDEX idx_subreddit ON raw_posts(subreddit);
```

---

## 📚 Subreddits Monitored

### Mobile & Wearables (6)
- r/apple, r/iphone, r/android, r/GooglePixel, r/samsung, r/GalaxyWatch

### Computers & Gaming (5)
- r/laptops, r/buildapc, r/pcgaming, r/pcmasterrace, r/battlestations

### Peripherals (3)
- r/mechanicalkeyboards, r/Monitors, r/headphones

### Gaming Handhelds (1)
- r/SteamDeck

### Smart Home (2)
- r/HomeAutomation, r/smarthome

### General & Support (3)
- r/technology, r/gadgets, r/TechSupport

---

## 🔬 Data Quality Measures

**Collection Filtering:**
- Minimum title length: 10 characters
- Excludes deleted/removed posts
- Excludes heavily downvoted content (score < -5)
- Spam detection and removal
- Duplicate prevention via database constraints

**Sentiment Analysis Quality:**
- Combined title + body text for context
- Handles empty posts gracefully
- Standard VADER thresholds (±0.05)
- Preserves emotional intensity (CAPS, emojis, punctuation)

**Current Quality Score:** 8/10 based on manual review

---

## 🚀 Week 7 Goal: RAG-Based Q&A System

### Planned Implementation

**Core Components:**
1. **Vector Embeddings**
   - Use sentence-transformers for semantic search
   - Convert all 17K+ posts to vector embeddings
   - Store in vector database (ChromaDB/Pinecone)

2. **Retrieval System**
   - Semantic search to find relevant posts
   - Rank by relevance to user query
   - Retrieve top 10-20 most relevant discussions

3. **LLM Integration**
   - Claude or GPT API for answer generation
   - Context-aware responses using retrieved posts
   - Source attribution and citations

4. **Chat Interface**
   - Natural language query input
   - Streaming responses
   - Display source posts with links
   - Conversation history

### Target Capability

**User Query:**
```
"Should I buy the Steam Deck or wait for Steam Deck 2?"
```

**System Response:**
```
Based on 347 recent discussions in r/SteamDeck, 73% of users 
recommend buying now. Here's why:

No Official Announcement: There's no confirmed Steam Deck 2 release 
date. Community consensus suggests it won't arrive until at least 
late 2026.

Current Generation Maturity: The original Steam Deck has received 
significant software improvements and has a large game compatibility 
library.

Strong Value: At $399, it's considered good value, especially with 
recent sales dropping it to $349.

Sources: 
- r/SteamDeck discussions (347 posts, Oct 2025)
- Average sentiment: +0.42 (positive)
- Top concerns: Battery life (mentioned 89 times)
```

### Architecture for RAG

```
User Question
    ↓
Vector Search (find relevant posts)
    ↓
Retrieve Context (top 20 posts)
    ↓
LLM (Claude/GPT) + Context
    ↓
Generated Answer + Sources
    ↓
Display to User
```

---

## 🔮 Future Enhancements (Post-Project)

**Beyond Week 7:**
- Multi-modal analysis (images, videos from posts)
- Real-time sentiment tracking dashboard
- Comparative product analysis
- Sentiment prediction models
- Mobile application
- Commercial deployment

---

## 🤝 Contributing

This is an academic project (CSE299), but suggestions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Sumayer Khan Sajid**  
- ID: 2221818642  
- Course: CSE299 - Junior Design Project  
- Semester: Fall 2025  
- University: North South University

---

## 🙏 Acknowledgments

- Reddit API (PRAW) for data access
- VADER Sentiment Analysis toolkit
- GitHub Actions for automation
- Open source community
- Course instructor and mentors

---

## 📧 Contact

- GitHub: [@SumayerKhan](https://github.com/SumayerKhan)
- Email: sumayer.cse.nsu@gmail.com

---

**⚠️ Note:** This project is for educational purposes. All data collection follows Reddit's Terms of Service and API usage guidelines. Database file is excluded from version control for size and privacy reasons.

---

**Last Updated:** October 26, 2025  
**Project Status:** Week 3 Complete - Sentiment Analysis Operational ✅