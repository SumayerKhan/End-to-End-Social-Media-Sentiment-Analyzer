# Data Collector Module

**Week 1-4 Implementation: Automated Reddit Data Collection**

This module collects Reddit posts from 20 consumer electronics subreddits using the Reddit API (PRAW), running automatically every 3 hours via GitHub Actions.

---

## PART 1: Beginner's Learning Guide

### What Does This Module Do?

Imagine you're researching what people think about the iPhone 15. You could manually visit r/iphone, r/apple, r/tech and read through thousands of posts... or you could use this automated collector to gather all that data for you while you sleep.

**This module:**
1. Connects to Reddit using their official API
2. Visits 20 technology subreddits every 3 hours
3. Collects new posts from each subreddit
4. Adds sentiment analysis (positive/negative/neutral)
5. Generates vector embeddings for semantic search
6. Stores everything in Supabase (cloud database)

**Result:** A constantly growing dataset of 30,000+ posts about consumer electronics, automatically updated without manual work.

---

### Why Automated Collection?

**The Challenge:**
Tech opinions change daily. A single data collection gives you a snapshot, but you miss:
- New product launches
- Bug reports and fixes
- Evolving opinions over time
- Fresh user experiences

**The Solution:**
GitHub Actions runs this collector every 3 hours automatically:
```
12:00 AM → Collect ~900 new posts
3:00 AM  → Collect ~900 new posts
6:00 AM  → Collect ~900 new posts
...
```

Result: ~7,200 new posts per day, zero manual effort.

---

### How Reddit API Works

**Think of it like a librarian:**
```
You: "Give me the 100 newest posts from r/iphone"
Reddit API: "Here you go!" [returns posts with metadata]
You: "Now give me the 50 hottest posts"
Reddit API: "Here they are!"
```

**Reddit provides multiple "feeds" per subreddit:**
- **new**: Most recent posts (sorted by time)
- **hot**: Trending posts (sorted by engagement)
- **rising**: Posts gaining traction quickly

We collect from all three to maximize coverage.

---

### The Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    COLLECTION PIPELINE                       │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ 1. COLLECT    │  │ 2. ENRICH     │  │ 3. STORE      │
│               │  │               │  │               │
│ github_       │  │ supabase_     │  │ Supabase      │
│ collector.py  │  │ pipeline.py   │  │ Database      │
│               │  │               │  │               │
│ - Reddit API  │→│ - Sentiment   │→│ - PostgreSQL  │
│ - 20 subs     │  │ - Embeddings  │  │ - pgvector    │
│ - Dedup       │  │ - Timestamps  │  │ - 32K+ posts  │
└───────────────┘  └───────────────┘  └───────────────┘
```

**Step 1 - Collect (`github_collector.py`):**
- Connects to Reddit API using PRAW
- Loops through 20 subreddits
- Fetches new/hot/rising posts from each
- Filters out invalid posts (deleted, too short, spam)
- Deduplicates (same post in multiple feeds)

**Step 2 - Enrich (`supabase_pipeline.py`):**
- Adds VADER sentiment scores
- Generates 384-dim embeddings for semantic search
- Converts timestamps to database format
- Prepares data for insertion

**Step 3 - Store:**
- Inserts to Supabase in batches of 100
- Skips duplicates (based on post_id)
- Reports success/failure statistics

---

### What Data Gets Collected?

**For each post, we capture:**

```python
{
    # Reddit metadata
    'post_id': 'abc123',                 # Unique identifier
    'subreddit': 'iphone',               # Which subreddit
    'title': 'iPhone 15 Pro battery...',  # Post title
    'selftext': 'Full post content...',   # Body text
    'author': 'username',                 # Author (not /u/deleted)
    'created_utc': '2025-11-02...',      # When posted
    'score': 245,                        # Upvotes - downvotes
    'num_comments': 89,                  # Number of comments
    'url': 'https://reddit.com/...',     # Post URL
    'permalink': '/r/iphone/...',        # Reddit permalink
    'collected_at': '2025-11-02...',     # When we collected it

    # Enrichments (added by pipeline)
    'sentiment_pos': 0.543,              # Positive score
    'sentiment_neg': 0.0,                # Negative score
    'sentiment_neu': 0.457,              # Neutral score
    'sentiment_compound': 0.873,         # Overall sentiment
    'sentiment_label': 'positive',       # positive/negative/neutral
    'embedding': [0.123, -0.456, ...],   # 384-dim vector
}
```

---

### Quality Filters

**We only collect posts that pass these checks:**

```python
def is_valid_post(post):
    """Quality filters"""
    # Must have meaningful title
    if len(post.title) < 10:
        return False

    # Not deleted/removed
    if post.selftext in ['[removed]', '[deleted]']:
        return False

    # Real author (not banned/deleted account)
    if post.author is None:
        return False

    # Not heavily downvoted spam
    if post.score < -5:
        return False

    return True
```

**Why these filters?**
- Short titles like "Help!" or "Question" aren't useful
- Deleted content can't be analyzed
- Heavily downvoted posts are usually spam/trolls

---

### The 20 Subreddits

**Mobile & Wearables:**
- r/apple, r/iphone, r/android, r/GooglePixel, r/samsung, r/GalaxyWatch

**Computers & Gaming:**
- r/laptops, r/buildapc, r/pcgaming, r/pcmasterrace, r/battlestations

**Peripherals:**
- r/mechanicalkeyboards, r/Monitors, r/headphones

**Gaming Handhelds:**
- r/SteamDeck

**Smart Home:**
- r/HomeAutomation, r/smarthome

**General & Support:**
- r/technology, r/gadgets, r/TechSupport

**Why these?**
- Cover all major consumer electronics categories
- Active communities (high post volume)
- English-language posts
- Tech-focused discussions

---

### GitHub Actions Automation

**How it works:**

1. **GitHub Actions** = Free automation service from GitHub
2. I created a workflow file (`.github/workflows/collect_data.yml`)
3. It runs every 3 hours: `schedule: - cron: '0 */3 * * *'`
4. Triggers the collection pipeline automatically
5. Logs results to GitHub Actions dashboard

**Benefits:**
- Zero cost (GitHub Actions is free for public repos)
- Runs 24/7 without my laptop being on
- Automatic retries if it fails
- Full logs for debugging

**Collection schedule:**
```
12:00 AM UTC → ~900 posts
 3:00 AM UTC → ~900 posts
 6:00 AM UTC → ~900 posts
 9:00 AM UTC → ~900 posts
12:00 PM UTC → ~900 posts
 3:00 PM UTC → ~900 posts
 6:00 PM UTC → ~900 posts
 9:00 PM UTC → ~900 posts

Total: ~7,200 new posts per day
```

---

### Deduplication Strategy

**Problem:**
Same post can appear in multiple feeds:
```
r/iphone → new feed → Post #123
r/iphone → hot feed → Post #123 (DUPLICATE!)
r/iphone → rising feed → Post #123 (DUPLICATE!)
```

**Solution 1 - Collection-time dedup:**
```python
seen_ids = set()
unique_posts = []

for post in all_posts:
    if post['post_id'] not in seen_ids:
        seen_ids.add(post['post_id'])
        unique_posts.append(post)
```

**Solution 2 - Database-level dedup:**
```sql
-- post_id is PRIMARY KEY
INSERT INTO reddit_posts (...)
ON CONFLICT (post_id) DO NOTHING
```

Result: Zero duplicate posts in database.

---

## PART 2: Technical Documentation

### Module Structure

```
collector/
├── __init__.py              # Package initialization
├── reddit_config.py         # Reddit API credentials
├── github_collector.py      # Core collection logic
├── supabase_pipeline.py     # Full pipeline orchestrator
├── continuous_collector.py  # Legacy continuous mode
└── scheduler.py             # Legacy local scheduler
```

---

### API Reference

#### `github_collector.py`

##### `collect_from_subreddit(reddit, subreddit_name)`

**Purpose:** Collect posts from a single subreddit across all feeds

**Signature:**
```python
def collect_from_subreddit(reddit, subreddit_name: str) -> List[Dict[str, Any]]
```

**Parameters:**
- `reddit` (praw.Reddit): PRAW Reddit instance
- `subreddit_name` (str): Subreddit name (without r/)

**Returns:** List[Dict[str, Any]] - List of post dictionaries

**Behavior:**
- Fetches from new (100), hot (50), rising (25) feeds
- Applies quality filters via `is_valid_post()`
- Returns raw posts (no sentiment/embeddings yet)

**Example:**
```python
from reddit_config import get_reddit_client
from collector.github_collector import collect_from_subreddit

reddit = get_reddit_client()
posts = collect_from_subreddit(reddit, 'iphone')
print(f"Collected {len(posts)} posts from r/iphone")

# Example output:
# [OK] r/iphone - new: 100 posts
# [OK] r/iphone - hot: 50 posts
# [OK] r/iphone - rising: 25 posts
# [TOTAL] r/iphone: 175 posts
```

**Code Location:** `collector/github_collector.py:50-120`

---

##### `is_valid_post(post)`

**Purpose:** Quality filter to exclude spam/invalid posts

**Signature:**
```python
def is_valid_post(post) -> bool
```

**Filters Applied:**
```python
# 1. Title must be meaningful (>= 10 characters)
if len(post.title) < 10:
    return False

# 2. Not deleted/removed content
if post.selftext in ['[removed]', '[deleted]']:
    return False

# 3. Has real author (not banned/deleted account)
if post.author is None:
    return False

# 4. Not heavily downvoted spam
if post.score < -5:
    return False

return True
```

**Code Location:** `collector/github_collector.py:20-48`

---

#### `supabase_pipeline.py`

##### `main()`

**Purpose:** Complete pipeline orchestrator

**Signature:**
```python
def main() -> None
```

**Pipeline Steps:**
```python
# 1. Initialize clients
reddit = get_reddit_client()
supabase = get_client()
analyzer = SentimentIntensityAnalyzer()
embedding_model = SentenceTransformer(EMBEDDING_MODEL)  # if ENABLE_EMBEDDINGS

# 2. Collect from all 20 subreddits
raw_posts = collect_all_posts(reddit)

# 3. Add sentiment analysis + convert timestamps
enriched_posts = enrich_posts_with_sentiment(raw_posts, analyzer)

# 4. Generate embeddings (optional)
if ENABLE_EMBEDDINGS:
    enriched_posts = enrich_posts_with_embeddings(
        posts=enriched_posts,
        model=embedding_model,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress=False
    )

# 5. Insert to Supabase
result = insert_posts_to_supabase(supabase, enriched_posts)

# 6. Print statistics
print(f"[OK] Inserted {result['success']:,} posts")
print(f"[INFO] {result['skipped']:,} posts skipped (duplicates)")
```

**Configuration:**
```python
BATCH_SIZE = 100              # Posts per database insert
ENABLE_EMBEDDINGS = True      # Toggle inline embeddings
```

**Output Example:**
```
[COLLECTION] Gathering posts from subreddits...
[OK] r/apple: 45 posts
[OK] r/iphone: 67 posts
...
[OK] Collected 3,458 posts total
[INFO] Removed 558 duplicate posts from same collection
[OK] 2,900 unique posts ready for insertion

[ENRICHMENT] Adding sentiment analysis...
[OK] Sentiment analysis complete

[ENRICHMENT] Generating embeddings...
[OK] Generated 2,900 embeddings

[INSERTION] Uploading 2,900 posts to Supabase...
[OK] Inserted 900 posts
[INFO] 2,000 posts skipped (legitimate duplicates from previous runs)

============================================================
FINAL STATISTICS
============================================================
[OK] New posts added to database: 900
[INFO] Duplicate posts skipped: 2,000
[TIME] Total pipeline time: 12m 34s
```

**Code Location:** `collector/supabase_pipeline.py:160-220`

---

##### `enrich_posts_with_sentiment(posts, analyzer)`

**Purpose:** Add VADER sentiment scores to posts

**Signature:**
```python
def enrich_posts_with_sentiment(
    posts: List[Dict[str, Any]],
    analyzer: SentimentIntensityAnalyzer
) -> List[Dict[str, Any]]
```

**Arguments:**
- `posts` (List[Dict]): Raw posts from collector
- `analyzer` (SentimentIntensityAnalyzer): VADER instance

**Returns:** List[Dict] - Posts with sentiment fields added

**Process:**
```python
for post in posts:
    # 1. Prepare text (combine title + body)
    text = prepare_text_for_sentiment(
        title=post['title'],
        body=post.get('selftext', '')
    )

    # 2. Calculate sentiment
    sentiment = calculate_sentiment(text, analyzer)
    # Returns: {
    #   'sentiment_pos': 0.234,
    #   'sentiment_neg': 0.056,
    #   'sentiment_neu': 0.710,
    #   'sentiment_compound': 0.632,
    #   'sentiment_label': 'positive'
    # }

    # 3. Convert timestamps (Unix → ISO 8601)
    post['created_utc'] = datetime.fromtimestamp(post['created_utc']).isoformat()
    post['collected_at'] = datetime.fromtimestamp(post['collected_at']).isoformat()

    # 4. Merge sentiment into post
    enriched_posts.append({**post, **sentiment})
```

**Code Location:** `collector/supabase_pipeline.py:59-96`

---

##### `collect_all_posts(reddit)`

**Purpose:** Collect from all configured subreddits with deduplication

**Signature:**
```python
def collect_all_posts(reddit) -> List[Dict[str, Any]]
```

**Arguments:**
- `reddit` (praw.Reddit): Reddit client instance

**Returns:** List[Dict] - Unique posts (deduplicated by post_id)

**Deduplication Logic:**
```python
seen_ids = set()
unique_posts = []
duplicates = 0

for post in all_posts:
    post_id = post.get('post_id')
    if post_id not in seen_ids:
        seen_ids.add(post_id)
        unique_posts.append(post)
    else:
        duplicates += 1

# Example: 3,458 total → 558 duplicates → 2,900 unique
```

**Output:**
```
[OK] Collected 3,458 posts total
[INFO] Removed 558 duplicate posts from same collection
[OK] 2,900 unique posts ready for insertion
```

**Code Location:** `collector/supabase_pipeline.py:99-135`

---

##### `insert_posts_to_supabase(supabase, posts)`

**Purpose:** Insert posts to Supabase in batches

**Signature:**
```python
def insert_posts_to_supabase(supabase, posts: List[Dict[str, Any]]) -> Dict[str, int]
```

**Arguments:**
- `supabase` (SupabaseClient): Supabase client instance
- `posts` (List[Dict]): Posts to insert

**Returns:** `{'success': int, 'skipped': int}`

**Behavior:**
- Inserts in batches of 100 (configurable via BATCH_SIZE)
- Uses UPSERT (insert or update if exists)
- Automatically skips duplicates (based on post_id PRIMARY KEY)

**Code Location:** `collector/supabase_pipeline.py:138-157`

---

### Configuration

**Reddit API credentials (`.env`):**
```bash
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=sentiment_analyzer/1.0
```

**Collection limits (`github_collector.py`):**
```python
FEED_LIMITS = {
    'new': 100,     # Most recent posts
    'hot': 50,      # Trending posts
    'rising': 25    # Growing fast
}
```

**Subreddits list:**
```python
SUBREDDITS = [
    'apple', 'iphone', 'android', ...  # 20 total
]
```

---

### Performance

**Collection speed:**
- ~20-30 seconds per subreddit (Reddit rate limits)
- ~10 minutes for all 20 subreddits
- Sentiment analysis: ~2-3 seconds for 1000 posts
- Embedding generation: ~30-60 seconds for 1000 posts
- Database insertion: ~5-10 seconds for 1000 posts

**Total pipeline time:** ~15-20 minutes per run

**Resource usage:**
- CPU: Moderate (embedding generation)
- Memory: ~200-500MB (model loading)
- Network: ~10-20MB download from Reddit
- Database: ~1-2MB per 1000 posts

---

### Error Handling

**Reddit API errors:**
```python
try:
    posts = collect_from_subreddit(reddit, subreddit_name)
except Exception as e:
    print(f"[ERROR] r/{subreddit_name}: {e}")
    # Continue with next subreddit
```

**Rate limiting:**
```python
time.sleep(2)  # Pause between subreddits
```

**Database errors:**
```python
try:
    supabase.insert_posts(batch)
except Exception as e:
    print(f"Error inserting batch: {e}")
    error_count += len(batch)
```

---

### Integration Points

**Imports from:**
- `reddit_config.py` → Reddit API client
- `analyzer.sentiment_utils` → Sentiment calculation
- `embeddings.embedding_utils` → Vector embeddings
- `supabase_db.db_client` → Database operations

**Used by:**
- GitHub Actions workflow (`.github/workflows/collect_data.yml`)
- Manual runs: `python collector/supabase_pipeline.py`

---

### Common Issues

**Issue 1: "Invalid credentials"**
- **Cause:** Missing/incorrect Reddit API keys
- **Fix:** Check `.env` file, verify credentials on Reddit

**Issue 2: "No new posts collected"**
- **Cause:** All posts already in database (duplicates)
- **Expected behavior:** Normal if running frequently

**Issue 3: "Rate limit exceeded"**
- **Cause:** Too many requests to Reddit
- **Fix:** Increase `time.sleep()` delays

---

### Monitoring

**Check collection status:**
```bash
python scripts/check_database.py
```

**Output:**
```
Total Posts:     32,595
Recent (3h):     900
Status:          ✅ HEALTHY - Automation working
```

**GitHub Actions logs:**
- Visit: https://github.com/[username]/[repo]/actions
- Check latest workflow run
- View logs for errors/statistics

---

**Last Updated:** November 2, 2025
**Module Status:** Production (running every 3 hours)
**Maintainer:** Sumayer Khan Sajid
