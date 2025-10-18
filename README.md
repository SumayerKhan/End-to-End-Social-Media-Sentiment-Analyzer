# End-to-End Social Media Sentiment Analyzer

> Automated system for collecting, analyzing, and visualizing public sentiment about consumer electronics from Reddit discussions

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Database](https://img.shields.io/badge/Database-SQLite-blue.svg)]()
[![Status](https://img.shields.io/badge/Status-Week%202%20Complete-green.svg)]()

---

## 🎯 Project Overview

A complete pipeline for real-time sentiment analysis of consumer electronics discussions on Reddit.

**System Components:**
- 🤖 **Continuous Collector** - Automated 24/7 data collection from 20 tech subreddits
- 🗄️ **Database** - SQLite storage with optimized schema and indexing
- 🔍 **API** - REST endpoints for keyword search and sentiment queries *(Coming Week 3)*
- 📊 **Dashboard** - Interactive Streamlit visualization *(Coming Week 5)*
- 🧠 **Sentiment Analysis** - VADER-based classification *(Coming Week 3)*

**Current Progress:** ✅ Data collection operational | 2,000+ posts collected

**Future Vision:** Upgrade to RAG-based Q&A system powered by LLMs for natural language queries about tech products

---

## 📈 Current Dataset Stats

- **Total Posts:** 2,066+ (and growing)
- **Subreddits Monitored:** 20 consumer electronics communities
- **Data Quality:** Filtered for spam, deleted content, and low-quality posts
- **Collection Method:** Official Reddit API (PRAW) - ethical and compliant
- **Update Frequency:** Continuous collection every 3 hours

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Reddit account (for API credentials)
- 500MB free disk space

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/SumayerKhan/End-to-End-Social-Media-Sentiment-Analyzer.git
cd sentiment-analyzer
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

### Running the Collector

**Single collection run:**
```bash
python collector/continuous_collector.py
```

**Check collected data:**
```bash
python database/check_db.py
```

**Preview data quality:**
```bash
python database/preview_data.py
```

---

## 📁 Project Structure
```
sentiment-analyzer/
├── collector/
│   ├── continuous_collector.py   # Main data collection script
│   ├── reddit_config.py          # Reddit API configuration
│   └── __init__.py
│
├── database/
│   ├── check_db.py               # Database statistics viewer
│   ├── preview_data.py           # Data quality checker
│   ├── tech_sentiment.db         # SQLite database (not in repo)
│   └── __init__.py
│
├── api/                          # REST API (Week 3)
├── analyzer/                     # Sentiment analysis (Week 3)
├── dashboard/                    # Streamlit UI (Week 5)
├── preprocessing/                # Text cleaning utilities
├── utils/                        # Helper functions
├── tests/                        # Unit tests
├── logs/                         # Log files (not in repo)
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
| **Language** | Python 3.9+ | Core development |
| **Reddit API** | PRAW 7.8.1 | Data collection |
| **Database** | SQLite3 | Local data storage |
| **Sentiment Analysis** | VADER | Text sentiment classification *(Week 3)* |
| **API Framework** | FastAPI | REST API endpoints *(Week 3)* |
| **Dashboard** | Streamlit | Interactive visualization *(Week 5)* |
| **Visualization** | Plotly | Charts and graphs *(Week 5)* |

---

## 🎓 Key Features

### ✅ Implemented (Weeks 1-2)
- [x] Multi-subreddit data collection (20 communities)
- [x] Three feed types per subreddit (new, hot, rising)
- [x] Quality filtering and spam detection
- [x] Automatic deduplication (INSERT OR IGNORE)
- [x] SQLite database with optimized indexes
- [x] Logging system for monitoring
- [x] Data verification tools

### 🔄 In Progress (Week 3)
- [ ] REST API with keyword search
- [ ] VADER sentiment analysis integration
- [ ] Result aggregation and statistics

### 📅 Planned (Weeks 4-7)
- [ ] Interactive Streamlit dashboard
- [ ] Time-series sentiment visualization
- [ ] Filtering by subreddit, date, sentiment
- [ ] Top posts by sentiment scores
- [ ] Testing and optimization

---

## 📊 Development Timeline

**Week 1-2: Data Collection** ✅
- Implemented Reddit API integration
- Built continuous collector with quality filters
- Created database schema with indexes
- Developed monitoring and verification tools

**Week 3-4: API & Sentiment Analysis** 🔄
- Build REST API with FastAPI
- Integrate VADER sentiment classifier
- Implement keyword search functionality
- Create aggregation endpoints

**Week 5-6: Dashboard & Visualization** 📅
- Develop Streamlit dashboard
- Create interactive sentiment charts
- Add filtering and search UI
- Implement data export features

**Week 7: Testing & Polish** 📅
- End-to-end testing
- Performance optimization
- Documentation completion
- Final presentation preparation

---

## 🗂️ Database Schema

### `raw_posts` Table
```sql
CREATE TABLE raw_posts (
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
    collected_at REAL NOT NULL
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

**Filtering Criteria:**
- Minimum title length: 10 characters
- Excludes deleted/removed posts
- Excludes heavily downvoted content (score < -5)
- Spam detection and removal
- Duplicate prevention via database constraints

**Current Quality Score:** 7/10 based on manual review

---

## 🚀 Future Enhancements (Post-Project)

### Phase 2: RAG System Upgrade
- **Vector Embeddings:** Semantic search using sentence-transformers
- **LLM Integration:** Claude/GPT API for natural language responses
- **RAG Pipeline:** Context-aware Q&A system
- **Enhanced UI:** Chat interface similar to Perplexity

**Target Capability:**
```
User: "Should I buy the Steam Deck or wait for Steam Deck 2?"

System: [Searches 10K+ relevant posts] → [Generates contextual answer]
"Based on 347 recent discussions in r/SteamDeck, 73% of users 
recommend buying now because Steam Deck 2 hasn't been announced 
and current generation is mature with good software support..."
```

---

## 🤝 Contributing

This is an academic project (CSE299), but suggestions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Sumayer Khan Sajid**  
- ID: 2221818642  
- Course: CSE299 - Junior Design Project  
- Semester: Fall 2025

---

## 🙏 Acknowledgments

- Reddit API (PRAW) for data access
- VADER Sentiment Analysis toolkit
- Open source community
- Course instructor and mentors

---

## 📧 Contact

- GitHub: [@SumayerKhan](https://github.com/SumayerKhan)
- Email: sumayer.cse.nsu@gmail.com

---

**⚠️ Note:** This project is for educational purposes. All data collection follows Reddit's Terms of Service and API usage guidelines.

---

**Last Updated:** October 19, 2025  
**Project Status:** Week 2 Complete - Data Collection Operational ✅