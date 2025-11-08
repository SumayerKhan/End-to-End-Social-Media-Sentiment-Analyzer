# Sentiment Analyzer Module

**Week 2-3 Implementation: VADER Sentiment Analysis**

This module analyzes the emotional tone of Reddit posts using VADER (Valence Aware Dictionary and sEntiment Reasoner), a lexicon-based sentiment analysis tool specifically designed for social media text.

---

## PART 1: Beginner's Learning Guide

### What is Sentiment Analysis?

Sentiment analysis is like having a robot read text and tell you whether it's happy, sad, or neutral.

**Real-world analogy:**
Imagine you're a product manager reading thousands of customer reviews. Your brain automatically notices:
- "This phone is amazing!" = Happy customer
- "Battery dies in 2 hours, worst purchase ever" = Angry customer
- "The phone arrived on time" = Neutral statement

Sentiment analysis automates this process. Instead of you reading 30,000 posts, a computer program reads them and labels each as positive, negative, or neutral.

---

### Why Do We Need Sentiment Analysis?

**The Problem:**
I collected 32,000+ Reddit posts about consumer electronics. Questions like:
- "What do people think about iPhone 15 battery life?"
- "Are gaming laptops worth it?"

Without sentiment analysis, the RAG system would just find posts mentioning these topics but couldn't tell you if people **love** or **hate** them.

**The Solution:**
Sentiment analysis adds emotional context:
- Post about "iPhone 15 battery" + **positive sentiment** = People are happy with it
- Post about "iPhone 15 battery" + **negative sentiment** = People have problems

This lets my RAG system answer: "Overall sentiment is positive, but some users report issues..."

---

### What is VADER?

**VADER** = Valence Aware Dictionary and sEntiment Reasoner

It's a pre-trained sentiment analysis tool that understands:
- **Intensity**: "good" vs "AMAZING!!!" (punctuation and caps matter)
- **Negation**: "not bad" is different from "bad"
- **Slang**: "sucks", "lit", "meh" (common in social media)
- **Emoticons**: :) vs :( vs :/

**Why VADER for Reddit?**
- Designed specifically for social media text
- Works great with informal language ("this phone slaps!")
- Fast (analyzes thousands of posts in seconds)
- Free and simple to use
- No training required (comes with pre-built dictionary)

**Alternatives I didn't choose:**
- **BERT/RoBERTa**: More accurate but 10-100x slower, requires GPU
- **OpenAI API**: Costs money for 30K posts
- **TextBlob**: Older, less accurate for social media

---

### How VADER Works (Simplified)

**Step 1: Look up each word in a sentiment dictionary**
```
"amazing" → +3.1 (very positive)
"good" → +1.9 (positive)
"okay" → +0.5 (slightly positive)
"bad" → -1.5 (negative)
"terrible" → -2.5 (very negative)
```

**Step 2: Apply modifiers**
```
"very good" → +1.9 * 1.3 = +2.47 (boost)
"not good" → +1.9 * -0.5 = -0.95 (flip to negative)
"AMAZING!!!" → +3.1 * 1.5 = +4.65 (emphasis)
```

**Step 3: Calculate overall scores**
For text: "The iPhone 15 is AMAZING! Battery life is great."

```
positive: 0.65  (65% of words are positive)
negative: 0.0   (0% of words are negative)
neutral: 0.35   (35% of words are neutral like "the", "is")
compound: 0.89  (overall score from -1 to +1)
```

**Step 4: Assign a label**
```
compound ≥ 0.05  → positive
compound ≤ -0.05 → negative
-0.05 < compound < 0.05 → neutral
```

---

### What This Module Does

This module takes Reddit posts and enriches them with sentiment scores:

**INPUT (raw post):**
```python
{
    'post_id': 'abc123',
    'title': 'iPhone 15 Pro battery life is amazing!',
    'selftext': 'I upgraded from iPhone 13 and it lasts all day',
    'subreddit': 'iphone'
}
```

**OUTPUT (post + sentiment):**
```python
{
    'post_id': 'abc123',
    'title': 'iPhone 15 Pro battery life is amazing!',
    'selftext': 'I upgraded from iPhone 13 and it lasts all day',
    'subreddit': 'iphone',
    'sentiment_pos': 0.543,        # 54.3% positive words
    'sentiment_neg': 0.0,          # 0% negative words
    'sentiment_neu': 0.457,        # 45.7% neutral words
    'sentiment_compound': 0.873,   # Overall: +0.873 (very positive)
    'sentiment_label': 'positive'  # Final label
}
```

Now the RAG system can:
- Filter: "Show me only positive reviews"
- Summarize: "60% positive, 30% neutral, 10% negative"
- Context: "Users are generally happy but..."

---

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  SENTIMENT ANALYZER                      │
└─────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Shared Utils │  │  Processor   │  │   Display    │
│              │  │              │  │              │
│ - calculate_ │  │ - process_   │  │ - show_      │
│   sentiment  │  │   posts_with │  │   results    │
│ - prepare_   │  │   _vader     │  │              │
│   text       │  │ - get_stats  │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
```

**Components:**

1. **sentiment_utils.py** - Core sentiment logic (reusable)
   - `calculate_sentiment()` - Run VADER on text
   - `prepare_text_for_sentiment()` - Combine title + body

2. **process_posts.py** - Batch sentiment analysis
   - Processes all posts in SQLite database
   - Shows progress bar and statistics
   - Used for Week 2-3 (before Supabase migration)

3. **show_results.py** - Visualize sentiment distribution
   - Print sentiment breakdown by subreddit
   - Display top positive/negative posts
   - Generate summary reports

**Note:** The sentiment utilities are now also used by:
- `collector/supabase_pipeline.py` (inline sentiment during collection)
- Future components that need sentiment analysis

---

### How My Pipeline Uses This Module

**Week 2-3 (Original SQLite approach):**
```
1. Collect posts → save to SQLite (database/)
2. Run analyzer/process_posts.py → add sentiment to all posts
3. Query database with sentiment filters
```

**Week 4+ (Current Supabase approach):**
```
1. Collect posts → calculate sentiment inline (using sentiment_utils)
2. Insert to Supabase with sentiment already attached
3. RAG retrieval can filter by sentiment_label
```

The analyzer module now serves two purposes:
- **Shared utilities** (`sentiment_utils.py`) used everywhere
- **Standalone tool** (`process_posts.py`) for batch processing

---

### Key Insights I Learned

**1. Text preparation matters:**
```python
# Bad: Only analyze title
"iPhone 15"  # neutral (no sentiment words)

# Good: Title + body combined
"iPhone 15" + "amazing battery life!" # positive!
```

**2. VADER is surprisingly good:**
- Handles "not bad" correctly (positive, not negative)
- Understands "!!!!!" adds emphasis
- Recognizes slang like "sucks" and "rocks"

**3. Sentiment distribution is realistic:**
```
Positive: ~48%  (people share good experiences)
Neutral:  ~32%  (questions, tech specs)
Negative: ~20%  (complaints, issues)
```

**4. Some posts are tricky:**
```
"Should I buy iPhone or Android?"  # neutral (question)
"iPhone 15 is okay, nothing special"  # neutral (lukewarm)
"The battery died after 2 hours"  # negative (bad experience)
```

---

### Usage Examples

**Example 1: Process all posts in database**
```python
from analyzer.process_posts import process_posts_with_vader

# Analyze all posts without sentiment scores
success = process_posts_with_vader()
```

**Example 2: Calculate sentiment for one post**
```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from analyzer.sentiment_utils import calculate_sentiment, prepare_text_for_sentiment

analyzer = SentimentIntensityAnalyzer()

# Prepare text
text = prepare_text_for_sentiment(
    title="iPhone 15 Pro battery life is amazing!",
    body="I upgraded from iPhone 13 and it lasts all day"
)

# Calculate sentiment
sentiment = calculate_sentiment(text, analyzer)

print(sentiment)
# Output:
# {
#     'sentiment_pos': 0.543,
#     'sentiment_neg': 0.0,
#     'sentiment_neu': 0.457,
#     'sentiment_compound': 0.873,
#     'sentiment_label': 'positive'
# }
```

**Example 3: Get sentiment statistics**
```python
from analyzer.process_posts import print_sentiment_report

# Print comprehensive report
print_sentiment_report()
```

---

## PART 2: Technical Documentation

### Module Structure

```
analyzer/
├── __init__.py               # Package initialization
├── sentiment_utils.py        # Core sentiment functions (shared)
├── process_posts.py          # Batch processing for SQLite
├── show_results.py           # Display and statistics
├── test_sentiment_analyzer.py  # Unit tests
└── add_sentiment_columns.py  # Database schema migration
```

---

### API Reference

#### `sentiment_utils.py`

##### `calculate_sentiment(text, analyzer)`

**Purpose:** Calculate VADER sentiment scores for text

**Signature:**
```python
def calculate_sentiment(text: str, analyzer: SentimentIntensityAnalyzer) -> Dict[str, Any]
```

**Parameters:**
- `text` (str): Text to analyze (title + body recommended)
- `analyzer` (SentimentIntensityAnalyzer): VADER analyzer instance

**Returns:**
```python
{
    'sentiment_pos': float,      # Positive score [0-1]
    'sentiment_neg': float,      # Negative score [0-1]
    'sentiment_neu': float,      # Neutral score [0-1]
    'sentiment_compound': float, # Compound score [-1 to 1]
    'sentiment_label': str       # 'positive', 'negative', or 'neutral'
}
```

**Algorithm:**
```python
# 1. Get VADER scores
scores = analyzer.polarity_scores(text)
# Returns: {'pos': 0.234, 'neg': 0.0, 'neu': 0.766, 'compound': 0.632}

# 2. Determine label based on compound score
compound = scores['compound']
if compound >= 0.05:
    label = 'positive'
elif compound <= -0.05:
    label = 'negative'
else:
    label = 'neutral'
```

**Label Thresholds:**
- `compound >= 0.05` → positive
- `compound <= -0.05` → negative
- `-0.05 < compound < 0.05` → neutral

**Example:**
```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from analyzer.sentiment_utils import calculate_sentiment

analyzer = SentimentIntensityAnalyzer()
sentiment = calculate_sentiment("This phone is amazing!", analyzer)
print(sentiment['sentiment_label'])  # "positive"
print(sentiment['sentiment_compound'])  # 0.632

# More examples:
calculate_sentiment("This is terrible", analyzer)
# → {'sentiment_label': 'negative', 'sentiment_compound': -0.5719, ...}

calculate_sentiment("This is okay", analyzer)
# → {'sentiment_label': 'neutral', 'sentiment_compound': 0.0, ...}

calculate_sentiment("NOT bad at all", analyzer)
# → {'sentiment_label': 'positive', 'sentiment_compound': 0.431, ...}  # Negation handled!
```

**Code Location:** `analyzer/sentiment_utils.py:10-50`

---

##### `prepare_text_for_sentiment(title, body)`

**Purpose:** Combine and prepare post text for sentiment analysis

**Signature:**
```python
def prepare_text_for_sentiment(title: str, body: str = None) -> str
```

**Parameters:**
- `title` (str): Post title (required)
- `body` (str, optional): Post body/selftext

**Returns:** str - Combined text string (title + body)

**Logic:**
```python
text = title or ""

# Add body if not empty
if body and body.strip():
    text += " " + body

return text.strip()
```

**Example:**
```python
from analyzer.sentiment_utils import prepare_text_for_sentiment

# With title and body
text = prepare_text_for_sentiment(
    title="iPhone 15 battery",
    body="Lasts all day with heavy use"
)
# Returns: "iPhone 15 battery Lasts all day with heavy use"

# With title only
text = prepare_text_for_sentiment(title="iPhone 15 battery")
# Returns: "iPhone 15 battery"

# With empty body (ignored)
text = prepare_text_for_sentiment(title="iPhone 15", body="")
# Returns: "iPhone 15"
```

**Code Location:** `analyzer/sentiment_utils.py:53-69`

**Used In:**
- `collector/supabase_pipeline.py:79` - Inline sentiment during collection
- `analyzer/process_posts.py` - Batch processing for SQLite

---

#### process_posts.py

**Main Functions:**

```python
def process_posts_with_vader(limit: int = None) -> bool
```
Process posts in SQLite database with VADER sentiment analysis.

**Parameters:**
- `limit` (int, optional): Number of posts to process (None = all)

**Returns:**
- `True` if successful, `False` if errors

**Behavior:**
- Fetches posts where `sentiment_compound IS NULL`
- Uses shared sentiment utilities for consistency
- Updates database with sentiment scores
- Shows progress every 100 posts
- Commits in batches for safety
- Prints summary statistics

**Example:**
```python
# Process all posts
success = process_posts_with_vader()

# Process first 1000 posts
success = process_posts_with_vader(limit=1000)
```

---

```python
def get_sentiment_statistics(detailed: bool = False) -> Dict[str, Any]
```
Get sentiment statistics from database.

**Parameters:**
- `detailed` (bool): Include breakdown by subreddit

**Returns:**
```python
{
    'total_posts': int,
    'posts_with_sentiment': int,
    'avg_compound': float,
    'avg_positive': float,
    'avg_negative': float,
    'avg_neutral': float,
    'coverage_percent': float,
    'label_distribution': {
        'positive': int,
        'negative': int,
        'neutral': int
    },
    'by_subreddit': [...]  # if detailed=True
}
```

**Example:**
```python
stats = get_sentiment_statistics(detailed=True)
print(f"Coverage: {stats['coverage_percent']:.1f}%")
print(f"Average sentiment: {stats['avg_compound']:.3f}")
```

---

```python
def print_sentiment_report() -> None
```
Print comprehensive sentiment analysis report.

**Output includes:**
- Overall statistics (total, coverage, averages)
- Sentiment distribution (positive/negative/neutral percentages)
- Breakdown by subreddit (top 15)

**Example:**
```python
print_sentiment_report()
# Prints formatted report to console
```

---

#### show_results.py

**Visualization Functions:**

```python
def show_sentiment_distribution() -> None
```
Display sentiment distribution across all posts.

---

### Database Schema

**Sentiment columns in `raw_posts` table (SQLite):**

```sql
CREATE TABLE raw_posts (
    -- ... other columns ...

    -- Sentiment scores (added by analyzer)
    sentiment_pos REAL,          -- Positive score [0-1]
    sentiment_neg REAL,          -- Negative score [0-1]
    sentiment_neu REAL,          -- Neutral score [0-1]
    sentiment_compound REAL,     -- Compound score [-1 to 1]
    sentiment_label TEXT         -- 'positive', 'negative', or 'neutral'
);

-- Index for filtering by sentiment
CREATE INDEX idx_sentiment_label ON raw_posts(sentiment_label);
CREATE INDEX idx_sentiment_compound ON raw_posts(sentiment_compound);
```

**Supabase schema:** Same columns, managed by `supabase_db/schema.sql`

---

### Configuration

**VADER Configuration** (built-in, no config needed):
- Positive threshold: `compound >= 0.05`
- Negative threshold: `compound <= -0.05`
- Neutral range: `-0.05 < compound < 0.05`

**Processing Configuration:**
```python
# In process_posts.py
DB_PATH = 'database/tech_sentiment.db'  # SQLite database path

# Progress updates every N posts
PROGRESS_INTERVAL = 100
```

---

### Performance

**Processing Speed:**
- ~500-1,000 posts/second on average CPU
- 30,000 posts → ~30-60 seconds total

**Memory Usage:**
- Processes in batches (commit every 100 posts)
- Low memory footprint (~50MB)

**Bottlenecks:**
- Database I/O (SQLite writes)
- Text concatenation

**Optimizations:**
- Shared utilities eliminate code duplication
- Batch commits reduce database overhead
- Progress indicators for user feedback

---

### Error Handling

**Empty text handling:**
```python
# If both title and body are empty, skip post
if not text_to_analyze:
    skipped += 1
    continue
```

**Database errors:**
```python
try:
    # Process posts
except Exception as e:
    print(f"[ERROR] Processing failed: {e}")
    traceback.print_exc()
    return False
```

**NULL safety:**
- Uses `WHERE sentiment_compound IS NULL` to avoid reprocessing
- Handles missing/deleted posts gracefully

---

### Testing

**Run tests:**
```bash
python analyzer/test_sentiment_analyzer.py
```

**Manual testing:**
```python
# Test sentiment calculation
from analyzer.sentiment_utils import calculate_sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# Test cases
test_cases = [
    "This is amazing!",           # Expected: positive
    "This is terrible",            # Expected: negative
    "This is okay",                # Expected: neutral
    "NOT bad at all",              # Expected: positive (negation)
    "This phone SUCKS!!!",         # Expected: negative (emphasis)
]

for text in test_cases:
    result = calculate_sentiment(text, analyzer)
    print(f"{text:30s} → {result['sentiment_label']:8s} ({result['sentiment_compound']:+.3f})")
```

---

### Integration Points

**Used by:**
1. **collector/supabase_pipeline.py** - Inline sentiment during collection
2. **analyzer/process_posts.py** - Batch processing for SQLite
3. **Future modules** - Any component needing sentiment analysis

**Provides:**
- Consistent sentiment calculation across entire project
- Reusable utilities for sentiment analysis
- Zero duplication (single source of truth)

---

### Common Issues

**Issue 1: All posts showing neutral**
- **Cause:** Only analyzing title (no sentiment words)
- **Fix:** Use `prepare_text_for_sentiment()` to include body

**Issue 2: Unexpected sentiment**
```python
"The battery is not bad"  # Shows positive (VADER handles negation)
"meh"                     # Shows neutral/negative (slang recognized)
```

**Issue 3: Processing already-processed posts**
- **Cause:** Missing `WHERE sentiment_compound IS NULL`
- **Fix:** Query checks for NULL before processing

---

### Migration Notes

**SQLite → Supabase:**
- Sentiment utilities (`sentiment_utils.py`) remain unchanged
- Same VADER configuration and thresholds
- Database schema identical (column names match)
- Process: SQLite uses `process_posts.py`, Supabase uses inline in pipeline

**Backwards compatibility:**
- `process_posts.py` still works for SQLite database
- Shared utilities ensure consistency across both approaches

---

### Future Enhancements

**Potential improvements:**
1. **Multi-aspect sentiment:** Analyze specific aspects (battery, camera, price)
2. **Emotion detection:** Beyond positive/negative (anger, joy, surprise)
3. **Fine-tuned BERT:** More accurate but slower (GPU required)
4. **Sentiment over time:** Track sentiment trends for products
5. **Subreddit-specific tuning:** Different thresholds per community

**Current limitations:**
- Single overall sentiment per post (not aspect-specific)
- English-only (VADER lexicon is English)
- No sarcasm detection (VADER limitation)

---

### References

**VADER Documentation:**
- Paper: https://ojs.aaai.org/index.php/ICWSM/article/view/14550
- GitHub: https://github.com/cjhutto/vaderSentiment
- PyPI: https://pypi.org/project/vaderSentiment/

**Related Modules:**
- `collector/` - Collects posts analyzed by this module
- `supabase_db/` - Stores sentiment scores
- `rag/` - Uses sentiment for filtering and context

---

**Last Updated:** November 2, 2025
**Module Status:** Production-ready (Week 2-3 complete)
**Maintainer:** Sumayer Khan Sajid
