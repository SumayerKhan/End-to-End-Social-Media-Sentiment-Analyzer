# End-to-End Social Media Sentiment Analyzer

> Automated system for collecting, analyzing, and visualizing public sentiment about consumer electronics from Reddit discussions

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Database](https://img.shields.io/badge/Database-Supabase-green.svg)](https://supabase.com)
[![Status](https://img.shields.io/badge/Status-Week%204%20Complete-brightgreen.svg)]()
[![RAG](https://img.shields.io/badge/RAG-Powered-purple.svg)]()

---

## 🎯 Project Overview

An intelligent RAG-based (Retrieval-Augmented Generation) Q&A system for analyzing consumer electronics sentiment from Reddit discussions. Ask natural language questions and get AI-powered insights backed by real community discussions.

**System Components:**
- 🤖 **Automated Collector** - GitHub Actions collects data every 3 hours ✅
- 🗄️ **Cloud Database** - Supabase (PostgreSQL + pgvector) ✅
- 🧠 **Sentiment Analysis** - VADER classification system ✅
- 🔄 **Automated Pipeline** - Fully automated data processing & sync ✅
- 🧬 **Vector Embeddings** - Semantic search with sentence-transformers *(Week 5)*
- 🤖 **RAG System** - LLM-powered question answering *(Week 5)*
- 💬 **Chat Interface** - Streamlit chat UI with source attribution *(Week 6)*
- 🚀 **Cloud Deployment** - Zero-cost hosting on Streamlit Cloud *(Week 7)*

**Current Progress:** ✅ Week 4 Complete | 📅 Week 5: RAG Pipeline Development

**Current Dataset:** 31,097+ posts analyzed with sentiment scores (growing ~2,000/day)

**Project Goal:** Production-ready RAG chatbot with automated data pipeline and cloud deployment

---

## 📈 Current Dataset Stats

- **Total Posts:** 31,097+ (and growing ~2,000/day)
- **Subreddits Monitored:** 20 consumer electronics communities
- **Sentiment Analysis:** Complete (VADER-based classification)
- **Growth Rate:** ~14,000 posts per week
- **Collection Method:** Official Reddit API (PRAW) - ethical and compliant
- **Update Frequency:** Automated collection every 3 hours via GitHub Actions
- **Database:** Migrating to Supabase for scalability

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
├── database/                     # Legacy SQLite utilities ✅
│   ├── tech_sentiment.db        # SQLite database (migration source)
│   ├── check_db.py             # Database statistics viewer
│   └── preview_data.py         # Data quality checker
│
├── analyzer/                     # Sentiment analysis (Week 3) ✅
│   ├── process_posts.py        # VADER sentiment processor
│   ├── show_results.py         # Results visualization
│   └── test_sentiment_analyzer.py # VADER testing utilities
│
├── supabase/                     # Cloud database (Week 4) 🔄
│   ├── migrate.py              # SQLite → Supabase migration
│   ├── schema.sql              # PostgreSQL schema with pgvector
│   ├── client.py               # Supabase client wrapper
│   └── sync_data.py            # Automated data sync from GitHub Actions
│
├── embeddings/                   # Vector embeddings (Week 4) 🔄
│   ├── generate_embeddings.py  # Create embeddings for all posts
│   ├── update_embeddings.py    # Incremental embedding updates
│   └── config.py               # Embedding model configuration
│
├── rag/                          # RAG pipeline (Week 5) 📅
│   ├── retriever.py            # Semantic search & ranking
│   ├── generator.py            # LLM integration (Groq API)
│   ├── pipeline.py             # Full RAG orchestration
│   └── prompts.py              # Prompt templates
│
├── chat/                         # Chat interface (Week 6) 📅
│   ├── app.py                  # Streamlit chat UI
│   ├── components.py           # UI components
│   └── chat_history.py         # Conversation management
│
├── scripts/                      # Automation scripts ✅
│   ├── auto_pipeline.py        # Complete pipeline orchestrator
│   └── import_from_github.py   # JSON importer (legacy)
│
├── data/collected/               # JSON files from GitHub Actions
│   └── reddit_posts_*.json     # Timestamped collections (100+ files)
│
├── .github/workflows/            # GitHub Actions automation ✅
│   ├── collect.yml             # Data collection (every 3 hours)
│   └── process_pipeline.yml    # Data processing & embeddings (every 6 hours)
│
├── .env                          # Secrets (NOT in repo)
├── .env.example                  # Environment template
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── LICENSE                       # MIT License
```

**Removed folders:** `api/`, `dashboard/`, `preprocessing/`, `utils/` (empty, not needed)

---

## 🛠️ Technical Stack (Zero-Cost Architecture)

| Component | Technology | Purpose | Cost |
|-----------|-----------|---------|------|
| **Language** | Python 3.11+ | Core development | Free |
| **Reddit API** | PRAW 7.8.1 | Data collection | Free |
| **Database** | Supabase (PostgreSQL) | Cloud data storage | Free (500MB) |
| **Vector Search** | pgvector | Semantic similarity search | Free (built-in) |
| **Embeddings** | sentence-transformers | Text-to-vector conversion | Free (open-source) |
| **Sentiment Analysis** | VADER | Text sentiment classification | Free |
| **LLM** | Groq API | Answer generation | Free (30 req/min) |
| **Chat UI** | Streamlit | Interactive chat interface | Free |
| **Automation** | GitHub Actions | Scheduled data processing | Free |
| **Hosting** | Streamlit Cloud | Web deployment | Free (1GB RAM) |

**Total Monthly Cost:** $0.00 (Free tier limits sufficient for university project)

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

### ✅ Week 4 Complete (Nov 1-7, 2025) - Cloud Migration
- [x] Migrated SQLite to Supabase (PostgreSQL + pgvector)
- [x] Set up automated GitHub Actions pipeline to Supabase
- [x] Fixed schema mismatches (feed_type, timestamp format)
- [x] Implemented deduplication to prevent batch conflicts
- [x] Database monitoring scripts (check_database.py)
- [x] Successfully collecting ~900 new posts every 3 hours

### 📅 Planned (Weeks 5-7 - RAG Development)
- [ ] **Week 5:** RAG retrieval system + Groq LLM integration
- [ ] **Week 6:** Streamlit chat interface with source attribution
- [ ] **Week 7:** Deployment, optimization, and presentation prep
- [ ] Advanced features: Multi-turn conversations, sentiment-weighted retrieval
- [ ] Zero-cost cloud deployment on Streamlit Cloud

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

**Week 1-2: Data Collection** ✅ COMPLETE
- Implemented Reddit API integration
- Built continuous collector with quality filters
- Created database schema with indexes
- Set up GitHub Actions automation
- Collected 31,097+ posts from 20 subreddits

**Week 3: Sentiment Analysis** ✅ COMPLETE
- Integrated VADER sentiment analyzer
- Added sentiment columns to database
- Processed all posts with sentiment scores
- Created automated pipeline script
- Built results visualization tools
- Completed documentation

**Week 4: Cloud Migration & Automation** ✅ COMPLETE
- Migrated SQLite → Supabase (PostgreSQL + pgvector)
- Set up automated GitHub Actions → Supabase pipeline
- Fixed schema compatibility (timestamps, deduplication)
- Built database monitoring tools
- Verified automation: ~900 new posts every 3 hours
- Database growing from 31K → 38K+ posts during week 4

**Week 5: RAG Pipeline** 📅 PLANNED
- Implement retrieval system (semantic + metadata filtering)
- Integrate Groq API for LLM responses
- Design prompt templates for accurate answers
- Build RAG orchestration pipeline
- Test answer quality and relevance

**Week 6: Chat Interface** 📅 PLANNED
- Develop Streamlit chat UI
- Add conversation history
- Implement source post attribution
- Create responsive UI components
- Deploy to Streamlit Cloud

**Week 7: Optimization & Presentation** 📅 PLANNED
- Optimize retrieval performance
- Improve prompt engineering
- Add advanced features (multi-turn, sentiment weighting)
- Create demo scenarios
- Prepare final presentation
- Polish documentation

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

## 🤖 RAG-Based Q&A System Architecture

### Implementation Stack

**Core Components:**
1. **Vector Embeddings (Week 4)**
   - Model: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
   - 31K+ posts converted to embeddings
   - Stored in Supabase with pgvector extension
   - Automated embedding generation for new posts

2. **Retrieval System (Week 5)**
   - Hybrid search: Semantic (pgvector) + metadata filtering
   - Filters: Subreddit, date range, sentiment score
   - Ranking by cosine similarity + recency + sentiment
   - Retrieve top 15-20 most relevant posts

3. **LLM Integration (Week 5)**
   - Groq API with Llama 3.2/Mixtral models
   - Context-aware responses using retrieved posts
   - Source attribution with Reddit permalinks
   - Sentiment-weighted answer generation

4. **Chat Interface (Week 6)**
   - Streamlit chat UI with message history
   - Real-time streaming responses (if supported)
   - Display source posts with metadata
   - Export conversations as markdown

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

### System Architecture

```
GitHub Actions (Every 3 hours)
    ↓
Collect Reddit posts via PRAW
    ↓
Store in Supabase (PostgreSQL)
    ↓
Run VADER sentiment analysis
    ↓
Generate embeddings (sentence-transformers)
    ↓
Store vectors in Supabase (pgvector)

────────────────────────────────────

User Question (Streamlit Chat)
    ↓
Embed query (same model)
    ↓
Vector Search in Supabase (pgvector)
    ↓
Filter by metadata (date, sentiment, subreddit)
    ↓
Retrieve top 15-20 relevant posts
    ↓
Send to Groq API (Llama 3.2)
    ↓
Generate answer with sources
    ↓
Display in Streamlit UI
```

**Deployment:**
- **Data Pipeline:** GitHub Actions (automated, cloud)
- **Database:** Supabase (500MB free tier)
- **Chat App:** Streamlit Cloud (1GB RAM free tier)
- **LLM:** Groq API (30 requests/min free tier)
- **Total Cost:** $0/month

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

**Last Updated:** November 2, 2025
**Project Status:** Week 4 Complete - Ready for RAG Development ✅
**Current Dataset:** 32,000+ posts in Supabase with automated sync (growing ~7,200/day)