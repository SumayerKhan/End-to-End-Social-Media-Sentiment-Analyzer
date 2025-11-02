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

**Current Progress:** ✅ Week 4 Complete (with automated embeddings) | 📅 Week 5: RAG Pipeline Development

**Current Dataset:** 38,000+ posts in Supabase with sentiment scores (growing ~7,200/day)

**Project Goal:** Production-ready RAG chatbot with automated data pipeline and cloud deployment

---

## 📈 Current Dataset Stats

- **Total Posts:** 38,000+ (and growing ~7,200/day)
- **Subreddits Monitored:** 20 consumer electronics communities
- **Sentiment Analysis:** Complete (VADER-based classification)
- **Growth Rate:** ~900 new posts every 3 hours (automated)
- **Collection Method:** Official Reddit API (PRAW) - ethical and compliant
- **Update Frequency:** Automated collection every 3 hours via GitHub Actions
- **Database:** Supabase (PostgreSQL + pgvector) - fully migrated and operational

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+ (required for modern dependencies)
- Reddit account (for API credentials)
- Supabase account (free tier)
- Git installed

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

4. **Configure environment variables**

   Create a `.env` file with the following:

```bash
# Reddit API (get from https://www.reddit.com/prefs/apps)
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=sentiment_analyzer/1.0

# Supabase (get from https://supabase.com/dashboard)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_anon_key
```

---

## 🔄 Running the System

### Check Database Status

**View current database statistics:**
```bash
python scripts/check_database.py
```

This will show:
- Total posts in Supabase
- Posts by subreddit
- Sentiment distribution
- Recent collection stats
- Database growth metrics

### Test Supabase Connection

**Verify your Supabase credentials:**
```bash
python supabase_db/test_connection.py
```

### Local Collection (Optional)

**Collect new posts manually:**
```bash
python collector/github_collector.py
```

This runs the same collection script used by GitHub Actions automation.

### Monitor Automation

The system is **fully automated** via GitHub Actions:
- Collection runs every 3 hours automatically
- Posts are inserted directly to Supabase
- VADER sentiment analysis runs automatically
- No manual intervention required

**View automation logs:**
- Check GitHub Actions tab in repository
- Monitor with `scripts/check_database.py`

---

## 📁 Project Structure
```
End-to-end sentiment analyzer/
│
├── collector/                         # Data collection (Weeks 1-2) ✅
│   ├── __init__.py
│   ├── github_collector.py           # GitHub Actions collector (automated)
│   ├── continuous_collector.py       # Alternative local collector
│   ├── supabase_pipeline.py          # Direct Supabase insertion pipeline
│   ├── reddit_config.py              # Reddit API configuration
│   └── scheduler.py                  # Collection scheduler
│
├── supabase_db/                       # Supabase database (Week 4) ✅
│   ├── __init__.py
│   ├── db_client.py                  # Supabase client wrapper with methods
│   ├── migrate.py                    # SQLite → Supabase migration script
│   ├── schema.sql                    # PostgreSQL schema with pgvector
│   └── test_connection.py            # Connection verification utility
│
├── embeddings/                        # Vector embeddings (Week 4) ✅
│   ├── __init__.py
│   ├── config.py                     # Embedding model configuration
│   ├── embedding_utils.py            # Shared embedding utilities (reusable)
│   └── generate_embeddings.py        # Batch embedding generation (backfilling)
│
├── analyzer/                          # Sentiment analysis (Week 3) ✅
│   ├── __init__.py
│   ├── process_posts.py              # VADER sentiment processor
│   ├── show_results.py               # Results visualization
│   └── test_sentiment_analyzer.py    # VADER testing utilities
│
├── database/                          # Legacy SQLite utilities (archived)
│   ├── tech_sentiment.db             # Original SQLite database
│   ├── check_db.py                   # SQLite statistics viewer
│   └── preview_data.py               # SQLite data quality checker
│
├── scripts/                           # Automation & monitoring ✅
│   ├── check_database.py             # Supabase database statistics
│   ├── log_database_size.py          # Database growth tracking
│   ├── auto_pipeline.py              # Legacy pipeline (SQLite)
│   └── import_from_github.py         # Legacy JSON importer
│
├── rag/                               # RAG pipeline (Week 5) 📅
│   └── (empty - to be implemented)
│
├── chat/                              # Chat interface (Week 6) 📅
│   └── (empty - to be implemented)
│
├── data/collected/                    # Legacy JSON files (archived)
│   └── reddit_posts_*.json           # Historical files (Oct 19 - Nov 2)
│
├── .github/workflows/                 # GitHub Actions automation ✅
│   ├── sync_to_supabase.yml          # Data collection → Supabase (every 3 hours)
│   └── collect.yml.disabled          # Legacy JSON collector (archived)
│
├── docs/                              # Documentation
│   └── (project documentation files)
│
├── tests/                             # Unit tests
│   └── (test files)
│
├── config/                            # Configuration files
│   └── (config files)
│
├── logs/                              # Application logs
│   └── (log files from local runs)
│
├── .env                               # Environment variables (NOT in repo)
├── .env.example                       # Environment variable template
├── .gitignore                         # Git ignore rules
├── requirements.txt                   # Python dependencies
├── CLAUDE.md                          # Project context for Claude Code
├── SUPABASE_MIGRATION_GUIDE.md        # Migration documentation
├── README.md                          # This file
└── LICENSE                            # MIT License
```

**Key Changes from Original Plan:**
- ✅ `supabase_db/` instead of `supabase/` (actual folder name)
- ✅ `collector/supabase_pipeline.py` - direct insertion to Supabase
- ✅ Week 4 complete: Database migration fully operational
- 📅 `rag/` and `chat/` folders exist but are empty (Week 5-6 work)
- 🗄️ Legacy SQLite files preserved in `database/` for reference

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

### ✅ Week 4 Complete (Nov 1-7, 2025) - Cloud Migration & Automated Pipeline
- [x] Migrated 31,097 posts from SQLite to Supabase (PostgreSQL + pgvector)
- [x] Set up automated GitHub Actions → Supabase direct insertion pipeline
- [x] Implemented automated VADER sentiment analysis in pipeline
- [x] **Added automated embedding generation (sentence-transformers)**
- [x] Created modular embedding utilities with zero code duplication
- [x] Created database monitoring and statistics tools
- [x] Achieved stable automation: ~900 new posts every 3 hours (~7,200/day)
- [x] Database grew from 31K to 38K+ posts during Week 4
- [x] Pipeline now: Collect → Sentiment → Embeddings → Insert

### 📅 Planned (Weeks 5-7 - RAG Development)
- [ ] **Week 5:** RAG retrieval system + Groq LLM integration (embeddings ready!)
- [ ] **Week 6:** Streamlit chat interface with source attribution
- [ ] **Week 7:** Deployment, optimization, and presentation prep
- [ ] Advanced features: Multi-turn conversations, sentiment-weighted retrieval
- [ ] Zero-cost cloud deployment on Streamlit Cloud

**Note:** Embeddings are now generated automatically during collection, so Week 5 can focus entirely on RAG retrieval and LLM integration!

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

## 🧬 Vector Embeddings Implementation

### sentence-transformers (all-MiniLM-L6-v2)

**Why This Model?**
- Fast inference on CPU (no GPU required)
- Small model size (~80MB download)
- 384-dimensional vectors (compact storage)
- Optimized for semantic similarity search
- Good balance of speed vs quality
- Perfect for RAG retrieval systems

**Technical Specifications:**
```python
Model: sentence-transformers/all-MiniLM-L6-v2
Vector Dimensions: 384
Processing Speed: ~1000 posts/minute on CPU
Storage: ~1.5KB per post (384 floats)
Similarity Metric: Cosine similarity
```

**Why Not Other Models?**
- `all-mpnet-base-v2`: Better quality but 768 dimensions (2x storage, slower)
- OpenAI `text-embedding-ada-002`: Costs money, requires API calls
- `all-distilroberta-v1`: Slower inference, 768 dimensions
- Custom training: No time/resources for university project

**Database Integration:**
```sql
-- Vector column in Supabase (pgvector extension)
embedding vector(384)

-- Fast similarity search index
CREATE INDEX ON reddit_posts USING ivfflat (embedding vector_cosine_ops);
```

**Embedding Pipeline (Week 5):**
1. Load pre-trained model from Hugging Face
2. Combine title + selftext for each post
3. Generate 384-dimensional embeddings
4. Store vectors in Supabase (pgvector column)
5. Create similarity search index
6. Enable semantic search for RAG queries

---

## 🔄 Automated Pipeline

### System Architecture (Current - Week 4 + Embeddings)

```
GitHub Actions (Cloud - Every 3 hours)
    ↓
1. Collect Reddit posts via PRAW
    ↓
2. Run VADER sentiment analysis
    ↓
3. Generate vector embeddings (sentence-transformers) ⭐ NEW
    ↓
4. Insert to Supabase (PostgreSQL)
    ↓
    [Supabase Cloud Database]
    ├─→ 38,000+ posts with sentiment scores + embeddings
    ├─→ Automated deduplication (PRIMARY KEY)
    ├─→ PostgreSQL with pgvector extension
    ├─→ 384-dimensional embeddings ready for RAG
    └─→ Growing ~900 posts every 3 hours
```

### Why This Architecture?

**Advantages:**
- ✅ Fully automated (zero manual intervention)
- ✅ Cloud-native (no local database management)
- ✅ Scalable (500MB free tier supports ~300K posts)
- ✅ Direct insertion (no intermediate JSON files)
- ✅ pgvector ready for semantic search
- ✅ Accessible from anywhere

**Key Files:**
- `collector/supabase_pipeline.py` - Runs in GitHub Actions (collection + sentiment + embeddings + insertion)
- `embeddings/embedding_utils.py` - Shared embedding utilities (zero duplication)
- `embeddings/generate_embeddings.py` - Batch embedding generation (for backfilling)
- `supabase_db/db_client.py` - Supabase client wrapper with helper methods
- `scripts/check_database.py` - Database monitoring and statistics
- `scripts/log_database_size.py` - Growth tracking (runs in GitHub Actions)

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

**Week 4: Cloud Migration & Automated Pipeline** ✅ COMPLETE
- Migrated 31,097 posts from SQLite to Supabase (PostgreSQL + pgvector)
- Set up automated GitHub Actions → Supabase direct insertion pipeline
- Implemented automated VADER sentiment analysis in collection pipeline
- **Added automated embedding generation with sentence-transformers**
- Created modular embedding utilities (embedding_utils.py)
- Built database monitoring and statistics tools (check_database.py)
- Achieved stable automation: ~900 new posts every 3 hours
- Database grew from 31K to 38K+ posts during Week 4
- Full pipeline: Collect → Sentiment → Embeddings → Insert

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
**Project Status:** Week 4 Complete (with automated embeddings) - Ready for RAG Development ✅
**Current Dataset:** 38,000+ posts in Supabase with sentiment + embeddings (growing ~7,200/day)
**Pipeline:** Collect → Sentiment → Embeddings → Insert (fully automated every 3 hours)