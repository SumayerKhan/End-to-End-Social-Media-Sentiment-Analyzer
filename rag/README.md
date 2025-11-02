# RAG Pipeline: A Complete Beginner's Guide

**Learning RAG (Retrieval-Augmented Generation) from First Principles**

---

## 📚 Table of Contents

1. [What is RAG? (And Why Do We Need It?)](#what-is-rag)
2. [The Problem RAG Solves](#the-problem-rag-solves)
3. [How RAG Works: The Big Picture](#how-rag-works-the-big-picture)
4. [Deep Dive: Understanding Each Component](#deep-dive-understanding-each-component)
5. [The Technology Stack Explained](#the-technology-stack-explained)
6. [How to Use This Implementation](#how-to-use-this-implementation)
7. [Configuration and Tuning](#configuration-and-tuning)
8. [Common Questions and Troubleshooting](#common-questions-and-troubleshooting)

---

## 📖 What is RAG?

### The Simple Explanation

Imagine you're writing an essay about a topic, but you can't remember all the details. What do you do?

1. **📚 Look up relevant information** in your textbooks and notes
2. **✍️ Write your essay** using the information you found
3. **📝 Cite your sources** so people can verify your claims

**RAG does exactly this, but with AI!**

- **R**etrieval: Find relevant information from a knowledge base
- **A**ugmented: Add that information to help the AI
- **G**eneration: AI writes an answer based on the retrieved information

### The Technical Definition

**RAG (Retrieval-Augmented Generation)** is an AI architecture that combines:
- **Information Retrieval** (like a search engine)
- **Natural Language Generation** (like ChatGPT)

This combination allows AI to answer questions using **specific, up-to-date information** from your own database, rather than relying only on what it learned during training.

---

## 🤔 The Problem RAG Solves

### Problem #1: LLMs Have Limited Knowledge

**Large Language Models (LLMs)** like GPT, Claude, or Llama are trained on general internet data. They:
- ❌ Don't know about YOUR specific data (your company docs, your Reddit posts, your research)
- ❌ Don't have information after their training cutoff date
- ❌ Can't access real-time or private information
- ❌ Sometimes "hallucinate" (make up plausible-sounding but wrong answers)

**Example Problem:**
```
You: "What do people on Reddit think about the iPhone 15 battery life?"
LLM without RAG: "I don't have access to recent Reddit discussions..."
```

### Problem #2: Fine-Tuning is Expensive and Inflexible

**Fine-tuning** (retraining an LLM on your data) is:
- 💰 Very expensive (thousands of dollars)
- ⏰ Time-consuming (days or weeks)
- 🔒 Frozen in time (doesn't update with new information)
- 🎯 Difficult to control what it learns

### The RAG Solution ✨

**RAG gives the LLM access to YOUR data without retraining:**
- ✅ Works with any LLM (no training needed)
- ✅ Updates instantly when you add new data
- ✅ Cost-effective (just API calls)
- ✅ Verifiable (provides source citations)
- ✅ Private (your data stays in your control)

**With RAG:**
```
You: "What do people on Reddit think about iPhone 15 battery life?"

System:
1. Searches your 32,000 Reddit posts
2. Finds 10 most relevant discussions about iPhone 15 battery
3. Feeds those posts to the LLM
4. LLM reads them and generates an answer with citations

LLM: "Based on Reddit discussions, users generally report positive
battery life [r/iphone, Post #3], though some experience rapid drain
after iOS updates [r/apple, Post #7]."
```

---

## 🎯 How RAG Works: The Big Picture

### The Three-Step Process

```
┌─────────────────────────────────────────────────────────────┐
│                    USER ASKS A QUESTION                      │
│            "What do people think about iPhone 15?"           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   STEP 1: RETRIEVAL                          │
│  Search your knowledge base for relevant information         │
│                                                              │
│  Input: "What do people think about iPhone 15?"              │
│  Search: 32,595 Reddit posts                                │
│  Output: Top 15 most relevant posts                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  STEP 2: AUGMENTATION                        │
│  Format the retrieved information as context for the LLM     │
│                                                              │
│  Take: Top 10 posts from those 15                           │
│  Format: Add metadata (subreddit, sentiment, score)          │
│  Build: Complete context with all relevant posts             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   STEP 3: GENERATION                         │
│  LLM reads the context and generates an answer               │
│                                                              │
│  LLM receives: Question + 10 relevant posts                  │
│  LLM generates: Answer based ONLY on those posts             │
│  LLM cites: Sources for verification                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     ANSWER WITH SOURCES                      │
│  "Based on Reddit discussions, users generally have          │
│   positive opinions about iPhone 15 [r/iphone, Post #1]..." │
└─────────────────────────────────────────────────────────────┘
```

### Why This Approach is Powerful

1. **Grounded Answers:** LLM can't hallucinate because it MUST use the provided context
2. **Verifiable:** Every claim has a source citation you can check
3. **Up-to-date:** As you add new data, the system finds it automatically
4. **Scalable:** Works with millions of documents
5. **Accurate:** Combines the search precision of retrieval with the language understanding of LLMs

---

## 🔬 Deep Dive: Understanding Each Component

### Component 1: The Embedder 🧮

**What it does:** Converts text into numbers (vectors) that capture meaning

#### The Problem It Solves

Computers can't understand text directly. They need numbers. But how do we convert "iPhone 15 has great battery life" into numbers that preserve its meaning?

#### The Solution: Embeddings

**Embeddings** are lists of numbers that represent the **semantic meaning** of text.

**Example:**
```python
text = "iPhone 15 battery life"
embedding = [0.23, -0.45, 0.67, 0.12, -0.89, ...]  # 384 numbers
```

**Key Insight:** Texts with similar meanings have similar embeddings!

```
"iPhone 15 battery"     → [0.23, -0.45, 0.67, ...]
"iPhone 15 battery life" → [0.24, -0.44, 0.68, ...]  ← Very similar!

"quantum physics theory" → [-0.67, 0.89, -0.12, ...] ← Very different!
```

#### How We Use It

**Model:** `sentence-transformers/all-MiniLM-L6-v2`
- Converts any text → 384-dimensional vector
- Trained on millions of text pairs
- Fast: ~0.1 seconds per query
- Free and open-source

**In Our Code (`embedder.py`):**
```python
from sentence_transformers import SentenceTransformer

# Load model once (cached for reuse)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert question to embedding
question = "What do people think about iPhone 15?"
embedding = model.encode(question)
# Returns: [0.234, -0.456, 0.789, ...] (384 numbers)
```

#### Why 384 Dimensions?

- **Too few dimensions (e.g., 50):** Can't capture nuanced meaning
- **Too many dimensions (e.g., 1536):** Slower and uses more storage
- **384 is the sweet spot:** Good quality, fast, efficient

**Think of it like describing a person:**
- 1 dimension: Just height (not enough!)
- 384 dimensions: Height, weight, hair color, eye color, personality traits... (captures the whole person!)

---

### Component 2: The Vector Database 🗄️

**What it does:** Stores embeddings and finds similar ones quickly

#### The Problem

You have 32,595 Reddit posts. Each has a 384-dimensional embedding. How do you find the posts most similar to your question **instantly**?

#### The Solution: Vector Similarity Search

**Technology:** Supabase PostgreSQL + pgvector extension

**How It Works:**

1. **Storage:**
   - Each Reddit post is stored with its embedding
   - Database table has a special `vector(384)` column

2. **Similarity Calculation:**
   - Use **cosine similarity** to compare vectors
   - Cosine similarity: measures the angle between vectors (0-1 scale)
   - 1 = identical, 0 = completely unrelated

3. **Fast Search:**
   - Special index (`ivfflat`) for quick similarity search
   - Can search millions of vectors in milliseconds

**Visual Explanation:**

```
Imagine embeddings as points in 384-dimensional space:

Your Question: ⭐ "iPhone 15 battery"

All Posts in Database:
📍 Post A: "iPhone 15 battery amazing" (very close to ⭐)
📍 Post B: "iPhone 15 camera quality" (somewhat close to ⭐)
📍 Post C: "Android phone review" (far from ⭐)
📍 Post D: "Laptop recommendations" (very far from ⭐)

Vector search finds the CLOSEST points = most relevant posts!
```

**In Our Code (`retriever.py`):**
```python
# Search for similar posts
posts = supabase.rpc('search_similar_posts', {
    'query_embedding': query_embedding,    # Your question as vector
    'match_threshold': 0.5,                 # Min similarity (0-1)
    'match_count': 15                       # Return top 15
})

# Returns posts sorted by similarity:
# Post #1: similarity 0.92 ⭐⭐⭐⭐⭐
# Post #2: similarity 0.87 ⭐⭐⭐⭐
# ...
# Post #15: similarity 0.52 ⭐⭐
```

#### Why Not Just Keyword Search?

**Keyword Search (Old Way):**
```
Query: "iPhone battery life"
Searches for posts containing: "iPhone" AND "battery" AND "life"

Problem: Misses posts like:
- "iPhone 15 Pro lasts all day" (no "battery" keyword!)
- "Apple phone power duration" (different words, same meaning!)
```

**Vector Search (RAG Way):**
```
Query embedding captures the MEANING: "smartphone battery performance"

Finds posts about:
✅ "iPhone battery lasting 2 days"
✅ "iPhone power consumption"
✅ "How long does iPhone last?"
✅ "iPhone energy efficiency"

All have similar MEANING, even with different WORDS!
```

---

### Component 3: The Context Builder 📋

**What it does:** Formats retrieved posts for the LLM to read

#### The Problem

You found 15 relevant posts. How do you present them to the LLM in a way that:
- Is clear and organized
- Includes important metadata
- Fits within the LLM's input limits
- Makes it easy for the LLM to cite sources

#### The Solution: Structured Context Formatting

**In Our Code (`prompt_templates.py`):**

```python
# Format each post clearly
POST #1
Source: r/iphone
Sentiment: POSITIVE
Relevance: 0.92
Reddit Score: 245 upvotes

Title: iPhone 15 Pro battery life is incredible
Content: Upgraded from iPhone 13, battery lasts all day with heavy use...

---

POST #2
Source: r/apple
Sentiment: NEGATIVE
Relevance: 0.87
Reddit Score: 89 upvotes

Title: Battery drain issue on iPhone 15
Content: Anyone else experiencing rapid battery drain after iOS update?

---
[... up to 10 posts total ...]

Total posts: 10
Sentiment distribution: 6 positive, 2 neutral, 2 negative
```

#### Why Format Like This?

1. **Numbered Posts:** Easy for LLM to cite (e.g., "[r/iphone, Post #1]")
2. **Metadata:** Gives LLM context about reliability (high upvotes = trusted)
3. **Sentiment:** Helps LLM understand opinion distribution
4. **Relevance Score:** Shows which posts are most important
5. **Summary:** Provides overview before details

#### Context Window Limits

**LLMs have limited input size:**
- GPT-4: ~8,000 tokens (~6,000 words)
- Llama-3.3-70b: ~8,000 tokens
- Mixtral: ~32,000 tokens

**Our Strategy:**
- Retrieve 15 posts (cast a wide net)
- Select top 10 (best quality/relevance ratio)
- Format concisely (fit within limits)

**Why only 10 posts when we have 32,595 in the database?**

Think of it like Google search:
- Google searches billions of pages
- Shows you top 10 results
- You read those 10, not all billions!

**RAG does the same:** Search all 32,595 → Show LLM the top 10 most relevant!

---

### Component 4: The LLM (Language Model) 🤖

**What it does:** Reads the context and generates a natural language answer

#### What is an LLM?

**LLM = Large Language Model**

It's an AI trained on massive amounts of text (books, websites, code) that can:
- Understand natural language
- Generate human-like text
- Follow instructions
- Reason about information
- Write in different styles

**Examples:** GPT-4, Claude, Llama, Mixtral

#### Our LLM Choice: Groq + Llama-3.3-70b

**Why Groq?**
- ✅ **Free tier:** 30 requests/minute (perfect for learning)
- ✅ **Fast:** Specialized hardware makes it 10x faster than normal
- ✅ **No credit card:** Sign up and start using immediately
- ✅ **Good models:** Access to Llama-3.3-70b, Mixtral, etc.

**Why Llama-3.3-70b?**
- ✅ **70 billion parameters:** Large enough for quality answers
- ✅ **Open source:** Meta's Llama model family
- ✅ **Well-tested:** Widely used and reliable
- ✅ **Good at instruction following:** Follows our citation requirements

#### How We Instruct the LLM

**System Prompt (Instructions):**
```
You are an expert analyst specializing in consumer electronics
sentiment analysis.

Answer questions using ONLY the Reddit posts provided.
Cite your sources as [r/subreddit, Post #X].
Provide balanced analysis with PROS and CONS.
For comparisons, give a clear verdict/recommendation.
```

**User Prompt (Question + Context):**
```
RELEVANT REDDIT DISCUSSIONS:
[All 10 formatted posts...]

USER QUESTION:
What do people think about iPhone 15 battery life?

Answer based on the discussions above. Cite sources.
Be helpful and actionable.
```

**In Our Code (`groq_client.py`):**
```python
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.3  # Lower = more focused/factual
)

answer = response.choices[0].message.content
```

#### Temperature: Controlling Creativity

**Temperature** controls how creative/random the LLM is:

```
Temperature = 0.1  (Very focused)
└─ Predictable, factual, repeatable
└─ Good for: Factual Q&A, data analysis
└─ Example: "Based on the data..."

Temperature = 0.7  (Balanced)
└─ Some variation, but still grounded
└─ Good for: General conversation, explanations
└─ Example: "From what users say..."

Temperature = 1.5  (Very creative)
└─ Unpredictable, imaginative, varied
└─ Good for: Creative writing, brainstorming
└─ Example: "Interestingly, there's a fascinating trend..."
```

**Our Setting:** Temperature = 0.3 (focused and factual, perfect for RAG!)

---

### Component 5: The Pipeline Orchestrator 🎵

**What it does:** Coordinates all components into a smooth workflow

#### The Challenge: Managing State

Each component needs expensive resources:
- **Embedder:** Model loading takes ~5 seconds
- **LLM Client:** API connection setup
- **Database Client:** Connection pooling

**Bad Approach (reload every time):**
```python
# ❌ SLOW! Loads model every single query
for question in 100_questions:
    model = load_embedding_model()     # 5 seconds each time!
    embedding = model.encode(question)
    # ...

# Total: 100 × 5 seconds = 500 seconds = 8+ minutes!
```

**Good Approach (load once, reuse):**
```python
# ✅ FAST! Load model once, reuse 100 times
pipeline = RAGPipeline()  # Loads model once (5 seconds)

for question in 100_questions:
    result = pipeline.query(question)  # Reuses loaded model (instant!)
    # ...

# Total: 5 seconds + (100 × 0.1s) = ~15 seconds!
```

#### The Class Design Pattern

**In Our Code (`pipeline.py`):**

```python
class RAGPipeline:
    def __init__(self):
        # Load resources ONCE (expensive!)
        self.embedding_model = load_model()      # ~5 seconds
        self.groq_client = Groq(api_key=...)    # ~0.2 seconds
        self.supabase = get_client()            # ~0.2 seconds
        # Total: ~5.4 seconds (one-time cost)

    def query(self, question):
        # Use pre-loaded resources (fast!)
        embedding = self.embedding_model.encode(question)  # 0.1s
        posts = self.supabase.search(embedding)            # 0.3s
        answer = self.groq_client.generate(posts)          # 2.0s
        # Total: ~2.4 seconds per query

        return answer
```

**This is the "Heated Oven" Pattern:**
- Heat oven once (initialization) = 5 seconds
- Bake many cookies (queries) = 2 seconds each
- Don't reheat oven for each cookie!

---

## 🛠️ The Technology Stack Explained

### Why These Specific Technologies?

Let's understand each technology choice:

#### 1. **sentence-transformers (Embedding Model)**

**What:** Python library for text embeddings
**Why:**
- ✅ Free and open-source
- ✅ Pre-trained models (no training needed!)
- ✅ Fast on CPU (works on any laptop)
- ✅ High quality results
- ✅ Easy to use (3 lines of code)

**Alternative Options:**
- OpenAI Embeddings: Better quality BUT costs money ❌
- Word2Vec: Older, lower quality ❌
- Custom training: Too complex, time-consuming ❌

#### 2. **Supabase (Database + Vector Search)**

**What:** PostgreSQL database with pgvector extension
**Why:**
- ✅ Free tier (500MB storage)
- ✅ Built-in vector search (pgvector)
- ✅ Real database (not just vector storage)
- ✅ Easy API (Python client)
- ✅ Cloud-hosted (no server management)

**Alternative Options:**
- Pinecone: Good but limited free tier ❌
- Weaviate: Complex setup ❌
- Chromadb: In-memory only (not persistent) ❌
- FAISS: Requires manual management ❌

#### 3. **Groq API (LLM Inference)**

**What:** Ultra-fast LLM API service
**Why:**
- ✅ 100% free tier (30 req/min)
- ✅ Fastest inference (specialized hardware)
- ✅ No credit card required
- ✅ Multiple models (Llama, Mixtral)
- ✅ Simple API (OpenAI-compatible)

**Alternative Options:**
- OpenAI GPT-4: Best quality BUT expensive ($0.01-0.03/query) ❌
- Anthropic Claude: Great BUT requires payment ❌
- Local Ollama: FREE BUT needs powerful GPU ❌
- Hugging Face Inference: Free BUT slow ❌

#### 4. **pgvector (Vector Similarity Search)**

**What:** PostgreSQL extension for vector operations
**Why:**
- ✅ Integrates with existing database
- ✅ ACID transactions (data integrity)
- ✅ Fast similarity search (ivfflat index)
- ✅ Supports cosine, L2, and inner product distance
- ✅ Mature and well-tested

**How It Works:**
```sql
-- Create table with vector column
CREATE TABLE reddit_posts (
    id TEXT PRIMARY KEY,
    title TEXT,
    embedding vector(384)  -- Special vector type!
);

-- Create similarity index for fast search
CREATE INDEX ON reddit_posts
USING ivfflat (embedding vector_cosine_ops);

-- Search for similar vectors
SELECT * FROM reddit_posts
ORDER BY embedding <=> query_vector  -- Cosine distance operator
LIMIT 10;
```

---

## 💻 How to Use This Implementation

### Prerequisites

**System Requirements:**
- Python 3.11 or higher
- Internet connection (for API calls)
- 2GB+ RAM (for embedding model)
- No GPU needed!

**Required Accounts (All Free):**
- Groq API account ([console.groq.com](https://console.groq.com))
- Supabase account (if running your own instance)

### Installation

**1. Clone the repository:**
```bash
cd "E:\Code\End-to-end sentiment analyzer"
```

**2. Activate virtual environment:**
```bash
# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables:**

Create/edit `.env` file:
```bash
# Groq API (for LLM)
GROQ_API_KEY=your_groq_api_key_here

# Supabase (for database)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your_service_key_here
```

### Quick Start (3 Ways)

#### Method 1: Interactive Mode (Recommended for Learning)

```bash
python rag/test_rag.py --mode interactive
```

**What you'll see:**
```
Initializing pipeline...
Pipeline ready!

Your question: What do people think about iPhone 15?
Thinking...

ANSWER:
Based on Reddit discussions, users generally have positive
opinions about iPhone 15...

Your question: Are gaming laptops worth it?
...

Your question: quit
Goodbye!
```

**Commands:**
- Type any question to get an answer
- `stats` - Show database statistics
- `help` - Show available commands
- `quit` - Exit

#### Method 2: Quick Single Query

```bash
python rag/test_rag.py --mode quick --question "What do people think about MacBooks?"
```

**When to use:**
- Testing a specific question
- Running from a script
- Quick checks

#### Method 3: Python Code

```python
from rag.pipeline import RAGPipeline

# Initialize (one-time ~5 seconds)
pipeline = RAGPipeline()

# Ask a question
result = pipeline.query("What do people think about gaming laptops?")

# Get the answer
print(result['answer'])

# Get sources
for i, source in enumerate(result['sources'], 1):
    print(f"{i}. r/{source['subreddit']}: {source['title']}")
```

### Understanding the Output

**Result Structure:**
```python
{
    'answer': "Generated answer with citations...",

    'sources': [
        {
            'post_id': '1abc',
            'subreddit': 'iphone',
            'title': 'iPhone 15 Pro battery life is incredible',
            'selftext': 'Full post text...',
            'sentiment_label': 'positive',
            'similarity': 0.92,  # How relevant (0-1)
            'score': 245         # Reddit upvotes
        },
        # ... up to 10 posts
    ],

    'metadata': {
        'posts_used': 10,
        'total_posts_retrieved': 15,
        'has_citations': True,
        'citations': ['[r/iphone, Post #1]', ...],
        'timing': {
            'embed_time': 0.15,
            'retrieve_time': 0.42,
            'generate_time': 2.31,
            'total_time': 2.88
        }
    }
}
```

---

## ⚙️ Configuration and Tuning

### Key Settings in `rag/config.py`

#### Retrieval Settings

```python
# How many posts to retrieve from database
DEFAULT_TOP_K = 15
# ⬆️ Increase (20-30) for broader search
# ⬇️ Decrease (10) for faster queries

# Minimum similarity score to include
MIN_SIMILARITY_THRESHOLD = 0.5
# ⬆️ Increase (0.6-0.7) for higher quality matches only
# ⬇️ Decrease (0.3-0.4) to include more posts

# How far back to search
DEFAULT_DATE_RANGE_DAYS = 365
# Change to 30 for recent posts only
# Change to 1000 for all-time search
```

#### Context Settings

```python
# How many posts to send to LLM
MAX_CONTEXT_POSTS = 10
# ⬆️ Increase (15-20) for more context BUT slower
# ⬇️ Decrease (5-8) for faster BUT less context

# Include metadata in context
INCLUDE_METADATA = True
# Set to False to save tokens (but lose context)
```

#### LLM Settings

```python
# Which Groq model to use
GROQ_MODEL = "llama-3.3-70b-versatile"
# Alternatives:
# - "mixtral-8x7b-32768" (longer context)
# - "llama-3.1-8b-instant" (faster, smaller)

# Creativity vs Factuality
TEMPERATURE = 0.3
# ⬆️ Increase (0.5-0.7) for more creative answers
# ⬇️ Decrease (0.1) for more factual/consistent

# Response length
MAX_TOKENS = 1024
# ⬆️ Increase (2048) for longer answers
# ⬇️ Decrease (512) for shorter answers
```

#### Response Style

```python
RESPONSE_STYLE = "balanced"
# Options:
# - "concise" (2-3 sentences)
# - "balanced" (4-6 sentences with structure)
# - "detailed" (comprehensive analysis)
```

### Performance Tuning

**For Faster Queries:**
```python
DEFAULT_TOP_K = 10           # Retrieve fewer posts
MAX_CONTEXT_POSTS = 5        # Send less to LLM
MAX_TOKENS = 512             # Shorter responses
```

**For Higher Quality:**
```python
DEFAULT_TOP_K = 30           # Cast wider net
MAX_CONTEXT_POSTS = 15       # More context for LLM
MIN_SIMILARITY_THRESHOLD = 0.6  # Only high-quality matches
TEMPERATURE = 0.2            # More factual
```

**For Specific Use Cases:**
```python
# Recent news/trends
DEFAULT_DATE_RANGE_DAYS = 30

# Historical analysis
DEFAULT_DATE_RANGE_DAYS = 1000

# Positive feedback only
# Use: sentiment_filter="positive" in query
```

---

## ❓ Common Questions and Troubleshooting

### Conceptual Questions

**Q: Why not just use ChatGPT?**

A: ChatGPT doesn't have access to YOUR data (your Reddit posts). RAG gives the LLM access to your specific knowledge base!

**Q: How is this different from fine-tuning?**

A:
- **Fine-tuning:** Retrain the entire model on your data ($$$, slow, frozen in time)
- **RAG:** Just provide relevant context at query time (free, instant, always up-to-date)

**Q: Can the LLM see all 32,595 posts?**

A: No! LLMs have input limits (~8,000 tokens). We search all 32,595 but only send the top 10 most relevant to the LLM.

**Q: Why embeddings instead of keyword search?**

A: Embeddings understand MEANING, not just words:
- Keyword: Misses "iPhone lasts all day" (no "battery" keyword)
- Embeddings: Finds it because the meaning is similar!

**Q: What if my question isn't in the database?**

A: The LLM will say "I don't have enough information" (honest answer!) based on the REQUIRE_SOURCE_CITATION setting.

### Technical Questions

**Q: Why does the first query take 5 seconds but later ones are faster?**

A: First query loads the embedding model (~5 sec). Subsequent queries reuse the loaded model (~2-3 sec).

**Q: Can I use a different LLM?**

A: Yes! Just change `GROQ_MODEL` in config.py to any Groq model, or modify `groq_client.py` to use OpenAI, Anthropic, etc.

**Q: How do I add more posts to the database?**

A: The automated GitHub Actions collector runs every 3 hours and adds ~900 new posts automatically!

**Q: What's the difference between top_k and MAX_CONTEXT_POSTS?**

A:
- `top_k`: How many posts to RETRIEVE from database (e.g., 15)
- `MAX_CONTEXT_POSTS`: How many to SEND to LLM (e.g., 10)
- We retrieve more, then select the best ones!

**Q: Why cosine similarity instead of Euclidean distance?**

A: Cosine measures angle (direction), Euclidean measures distance (magnitude). For text, direction matters more than magnitude!

### Troubleshooting

**Problem: "GROQ_API_KEY not found"**

Solution:
1. Go to https://console.groq.com/
2. Sign up (free, no credit card)
3. Generate API key
4. Add to `.env` file: `GROQ_API_KEY=gsk_...`

**Problem: "No relevant posts found"**

Solutions:
- Lower similarity threshold: `similarity_threshold=0.3`
- Increase posts retrieved: `top_k=30`
- Try broader questions
- Check database has embeddings: `python scripts/check_database.py`

**Problem: "Rate limit exceeded"**

Solution:
- Groq free tier: 30 requests/minute
- Add delays: `time.sleep(2)` between queries
- Upgrade to paid tier for more requests

**Problem: "Model loading is slow"**

This is normal! First load takes ~5 seconds. Solutions:
- Use the `RAGPipeline()` class to load once
- Subsequent queries will be fast
- Can't avoid initial model loading (it's a 80MB download)

**Problem: "Answers are too generic/not specific"**

Solutions:
- Increase `MAX_CONTEXT_POSTS` to 15
- Lower `similarity_threshold` to 0.4
- Increase `TEMPERATURE` slightly (0.4-0.5)
- Add more specific filters (subreddit, date)

---

## 📚 Further Learning

### Recommended Reading

**RAG Fundamentals:**
- [RAG Paper (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401) - Original research
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/) - Practical guide

**Vector Embeddings:**
- [Sentence-BERT Paper](https://arxiv.org/abs/1908.10084) - How our embedding model works
- [Hugging Face Embeddings Guide](https://huggingface.co/blog/getting-started-with-embeddings)

**Vector Databases:**
- [pgvector Documentation](https://github.com/pgvector/pgvector) - PostgreSQL vector extension
- [Vector Database Comparison](https://www.pinecone.io/learn/vector-database/)

### Experiment and Learn

**Modify and Test:**
1. Change the system prompt - make it give shorter/longer answers
2. Try different similarity thresholds - see how results change
3. Increase MAX_CONTEXT_POSTS to 20 - does quality improve?
4. Lower TEMPERATURE to 0.1 - how does style change?

**Build Your Own:**
- Replace Reddit data with your own documents
- Try different embedding models
- Experiment with different LLMs
- Add new features (summarization, Q&A, etc.)

---

## 🎯 Project Status

**Current Implementation:**
- ✅ Complete RAG pipeline (retrieval + generation)
- ✅ 32,595+ Reddit posts indexed
- ✅ Automated data collection (every 3 hours)
- ✅ Vector similarity search with filters
- ✅ Source citation in all answers
- ✅ Three response styles
- ✅ Interactive testing mode
- ✅ Comprehensive documentation

**Next Steps (Week 6):**
- 🎨 Streamlit chat interface
- 💬 Conversation history
- 🌐 Deploy to Streamlit Cloud
- 📊 Analytics dashboard

**Learning Outcomes:**
By studying this implementation, you've learned:
- ✅ What RAG is and why it's useful
- ✅ How embeddings capture semantic meaning
- ✅ How vector databases enable similarity search
- ✅ How to combine retrieval with LLM generation
- ✅ How to build a production-ready RAG system
- ✅ How to optimize for quality and performance

---

**Last Updated:** November 2, 2025
**Version:** 2.0 (Beginner-Friendly Learning Guide)
**Status:** ✅ Production-Ready, Fully Documented

---

# 📖 PART 2: Technical Documentation

**For developers who want detailed API references and implementation details**

---

## 📁 Module Reference

### `rag/config.py` - Configuration Management

**Purpose:** Central configuration for all RAG components

**Key Constants:**

```python
# Embedding Settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
MAX_TEXT_LENGTH = 512

# Retrieval Settings
DEFAULT_TOP_K = 15                    # Posts to retrieve
MIN_SIMILARITY_THRESHOLD = 0.5        # Min cosine similarity
DEFAULT_DATE_RANGE_DAYS = 365         # Search window

# LLM Settings
GROQ_MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0.3                     # 0-2, lower = more focused
MAX_TOKENS = 1024                     # Max response length
TOP_P = 0.9                           # Nucleus sampling

# Context Settings
MAX_CONTEXT_POSTS = 10                # Posts sent to LLM
INCLUDE_METADATA = True               # Include post metadata
REQUIRE_SOURCE_CITATION = True        # Force citations

# Response Settings
RESPONSE_STYLE = "balanced"           # concise/balanced/detailed
ENABLE_STREAMING = False              # Stream responses
VERBOSE = True                        # Print debug info
DEBUG_MODE = False                    # Show full prompts
```

**Functions:**

```python
def validate_config() -> bool:
    """
    Validates all configuration settings
    Raises ValueError if invalid
    Returns True if valid
    """
```

---

### `rag/embedder.py` - Query Embedding

**Purpose:** Converts text queries into vector embeddings

**Main Functions:**

```python
@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load and cache the embedding model

    Returns:
        SentenceTransformer: Cached model instance

    Note: Uses lru_cache for singleton pattern
    """

def embed_query(query: str) -> List[float]:
    """
    Convert a single query into an embedding vector

    Args:
        query (str): User's question

    Returns:
        List[float]: 384-dimensional embedding vector

    Example:
        >>> embedding = embed_query("What do people think about iPhone?")
        >>> len(embedding)
        384
    """

def embed_queries_batch(
    queries: List[str],
    batch_size: int = 32
) -> List[List[float]]:
    """
    Convert multiple queries into embeddings efficiently

    Args:
        queries: List of question strings
        batch_size: Number of queries to process at once

    Returns:
        List of embedding vectors
    """

def validate_embedding(embedding: List[float]) -> bool:
    """
    Validate embedding format and dimensions

    Args:
        embedding: Vector to validate

    Returns:
        True if valid

    Raises:
        ValueError: If dimension mismatch or invalid format
    """

def compute_similarity(
    embedding1: List[float],
    embedding2: List[float]
) -> float:
    """
    Compute cosine similarity between two embeddings

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        float: Similarity score (0-1, higher = more similar)
    """
```

**Performance:**
- First call: ~5 seconds (model loading)
- Subsequent calls: ~0.1 seconds (model cached)
- Batch processing: ~10ms per query in batches of 32

---

### `rag/retriever.py` - Vector Search

**Purpose:** Searches Supabase for semantically similar posts

**Main Functions:**

```python
def retrieve_similar_posts(
    query_embedding: List[float],
    top_k: int = DEFAULT_TOP_K,
    similarity_threshold: float = MIN_SIMILARITY_THRESHOLD,
    subreddit_filter: Optional[str] = None,
    sentiment_filter: Optional[str] = None,
    days_ago: int = DEFAULT_DATE_RANGE_DAYS
) -> List[Dict[str, Any]]:
    """
    Retrieve posts similar to the query embedding

    Args:
        query_embedding: Query vector (384 dimensions)
        top_k: Number of posts to retrieve
        similarity_threshold: Min cosine similarity (0-1)
        subreddit_filter: Filter by subreddit (e.g., "iphone")
        sentiment_filter: Filter by sentiment (positive/negative/neutral)
        days_ago: Only search posts from last N days

    Returns:
        List of post dictionaries with similarity scores

    Example:
        >>> posts = retrieve_similar_posts(
        ...     query_embedding=embedding,
        ...     top_k=10,
        ...     similarity_threshold=0.6,
        ...     subreddit_filter="iphone"
        ... )
        >>> posts[0]['similarity']
        0.87
    """

def rerank_by_relevance(
    posts: List[Dict[str, Any]],
    boost_score_weight: float = 0.1,
    boost_comments_weight: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Re-rank posts by combining similarity with engagement metrics

    Combines:
    - Vector similarity (main factor)
    - Reddit score/upvotes (minor boost)
    - Number of comments (minor boost)

    Args:
        posts: Posts with similarity scores
        boost_score_weight: Weight for Reddit score (0-1)
        boost_comments_weight: Weight for comment count (0-1)

    Returns:
        Re-ranked posts
    """

def get_diverse_posts(
    posts: List[Dict[str, Any]],
    max_per_subreddit: int = 3
) -> List[Dict[str, Any]]:
    """
    Ensure diversity across subreddits

    Args:
        posts: List of posts
        max_per_subreddit: Max posts per subreddit

    Returns:
        Diversified list of posts
    """
```

**Performance:**
- Vector search: ~0.2-0.5 seconds (32K posts)
- Filtering: Negligible overhead
- Index type: ivfflat with cosine similarity

---

### `rag/groq_client.py` - LLM API Client

**Purpose:** Handles communication with Groq LLM API

**Main Functions:**

```python
@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    """
    Get or create Groq API client (cached)

    Returns:
        Groq: Cached client instance
    """

def generate_completion(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    top_p: float = TOP_P,
    model: str = GROQ_MODEL
) -> str:
    """
    Generate a completion from Groq API

    Args:
        prompt: User prompt/question
        system_prompt: System instructions (optional)
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum response length
        top_p: Nucleus sampling parameter (0-1)
        model: Groq model name

    Returns:
        str: Generated text response

    Raises:
        Exception: If API call fails after MAX_RETRIES

    Example:
        >>> response = generate_completion(
        ...     prompt="What is RAG?",
        ...     system_prompt="You are a helpful assistant.",
        ...     temperature=0.3
        ... )
    """

def generate_completion_streaming(
    prompt: str,
    system_prompt: Optional[str] = None,
    **kwargs
) -> Generator[str, None, None]:
    """
    Generate a streaming completion from Groq API

    Args:
        prompt: User prompt
        system_prompt: System instructions
        **kwargs: Additional parameters

    Yields:
        str: Text chunks as they arrive

    Example:
        >>> for chunk in generate_completion_streaming("Tell me about AI"):
        ...     print(chunk, end='', flush=True)
    """

def test_api_connection() -> bool:
    """
    Test Groq API connection

    Returns:
        bool: True if connection successful
    """
```

**Error Handling:**
- Automatic retry: 3 attempts with 2-second delays
- Rate limiting: 30 requests/minute (free tier)
- Connection errors: Graceful fallback

---

### `rag/prompt_templates.py` - Prompt Engineering

**Purpose:** System prompts and context formatting

**Main Functions:**

```python
def get_system_prompt(style: str = RESPONSE_STYLE) -> str:
    """
    Get the system prompt for the RAG pipeline

    Args:
        style: Response style (concise/balanced/detailed)

    Returns:
        str: System prompt with instructions

    Notes:
        - Instructs LLM on citation format
        - Emphasizes balanced analysis with PROS/CONS
        - Includes comparison-specific instructions
    """

def build_context_from_posts(
    posts: List[Dict[str, Any]],
    max_posts: int = MAX_CONTEXT_POSTS
) -> str:
    """
    Build the full context section from retrieved posts

    Args:
        posts: Retrieved posts
        max_posts: Maximum posts to include

    Returns:
        str: Formatted context string

    Format:
        --- POST #1 ---
        Source: r/subreddit
        Sentiment: POSITIVE
        Relevance Score: 0.92
        Reddit Score: 245 upvotes

        Title: Post title
        Content: Post content...
    """

def format_user_prompt(question: str, context: str) -> str:
    """
    Format the user prompt with question and context

    Args:
        question: User's question
        context: Formatted context from posts

    Returns:
        str: Complete user prompt

    Features:
        - Detects comparison questions automatically
        - Adds specific instructions for comparisons
        - Includes citation requirements
    """

def validate_response_has_citations(response: str) -> bool:
    """
    Check if response contains source citations

    Args:
        response: LLM response

    Returns:
        bool: True if citations found

    Citation format: [r/subreddit, Post #X]
    """

def extract_cited_posts(response: str) -> List[str]:
    """
    Extract all cited post references from response

    Args:
        response: LLM response with citations

    Returns:
        List[str]: List of unique citations
    """
```

**Prompt Structure:**

```
SYSTEM PROMPT:
- Role definition (expert analyst)
- Instructions (cite sources, balanced analysis)
- Comparison guidelines (PROS/CONS/Verdict)
- Style instructions (concise/balanced/detailed)

USER PROMPT:
- Formatted context (10 posts with metadata)
- User question
- Comparison detection (if applicable)
- Citation requirements
```

---

### `rag/generator.py` - Response Generation

**Purpose:** Combines retrieval with LLM to generate answers

**Main Functions:**

```python
def generate_answer(
    question: str,
    retrieved_posts: List[Dict[str, Any]],
    style: str = RESPONSE_STYLE,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    max_context_posts: int = MAX_CONTEXT_POSTS
) -> Dict[str, Any]:
    """
    Generate an answer to a question using retrieved posts

    Args:
        question: User's question
        retrieved_posts: Posts from vector search
        style: Response style
        temperature: LLM temperature
        max_tokens: Max response length
        max_context_posts: Max posts in context

    Returns:
        Dict with:
        - answer: Generated answer text
        - sources: List of source posts used
        - metadata: Additional information

    Example:
        >>> result = generate_answer(
        ...     "What do people think about iPhone?",
        ...     retrieved_posts=posts
        ... )
        >>> print(result['answer'])
        >>> print(f"Used {len(result['sources'])} sources")
    """

def generate_answer_with_sources_formatted(
    question: str,
    retrieved_posts: List[Dict[str, Any]],
    **kwargs
) -> str:
    """
    Generate answer and format with sources section

    Args:
        question: User's question
        retrieved_posts: Retrieved posts
        **kwargs: Additional arguments

    Returns:
        str: Formatted string with answer + sources + metadata
    """

def validate_answer_quality(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the quality of a generated answer

    Args:
        result: Result from generate_answer()

    Returns:
        Dict with:
        - valid: bool
        - issues: List[str]
        - score: float (0-1)

    Checks:
    - Answer length (not too short/long)
    - Citation presence (if required)
    - Error indicators
    - Source usage
    """
```

---

### `rag/pipeline.py` - Main RAG Orchestrator

**Purpose:** Coordinates all components into a complete pipeline

**Class: RAGPipeline**

```python
class RAGPipeline:
    """
    Main RAG pipeline that orchestrates all components

    Attributes:
        embedding_model: Cached SentenceTransformer model
        groq_client: Cached Groq API client
        supabase_client: Cached Supabase client
        verbose: Whether to print debug info

    Example:
        >>> pipeline = RAGPipeline()  # Loads models (~5 seconds)
        >>> result = pipeline.query("What do people think about iPhone?")
        >>> print(result['answer'])
    """

    def __init__(self, verbose: bool = VERBOSE):
        """
        Initialize RAG pipeline

        Loads and caches:
        - Embedding model (~5 seconds)
        - Groq LLM client (~0.2 seconds)
        - Supabase database client (~0.2 seconds)

        Args:
            verbose: Print initialization progress
        """

    def query(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = MIN_SIMILARITY_THRESHOLD,
        subreddit_filter: Optional[str] = None,
        sentiment_filter: Optional[str] = None,
        days_ago: int = DEFAULT_DATE_RANGE_DAYS,
        style: str = RESPONSE_STYLE,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        rerank: bool = False,
        diversify: bool = False
    ) -> Dict[str, Any]:
        """
        Query the RAG pipeline with a question

        Args:
            question: User's question
            top_k: Number of posts to retrieve
            similarity_threshold: Min similarity (0-1)
            subreddit_filter: Filter by subreddit
            sentiment_filter: Filter by sentiment
            days_ago: Search window in days
            style: Response style
            temperature: LLM temperature
            max_tokens: Max response length
            rerank: Re-rank by engagement metrics
            diversify: Ensure subreddit diversity

        Returns:
            Dict with:
            - answer: str
            - sources: List[Dict]
            - metadata: Dict (timing, citations, etc.)

        Performance:
            - First query: ~7 seconds (includes model loading)
            - Subsequent: ~3 seconds
        """

    def query_formatted(self, question: str, **kwargs) -> str:
        """
        Query and return formatted output

        Args:
            question: User's question
            **kwargs: Query parameters

        Returns:
            str: Formatted answer with sources
        """

    def test_connection(self) -> bool:
        """
        Test all connections

        Returns:
            bool: True if all tests pass
        """

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics

        Returns:
            Dict with database stats
        """
```

**Convenience Functions:**

```python
def get_pipeline(verbose: bool = VERBOSE) -> RAGPipeline:
    """
    Get or create singleton pipeline instance

    Returns:
        RAGPipeline: Singleton instance
    """

def quick_query(question: str, **kwargs) -> str:
    """
    Quick query function for simple use cases

    Args:
        question: User's question
        **kwargs: Query parameters

    Returns:
        str: Formatted answer with sources
    """
```

---

## 🗄️ Database Schema

### Table: `reddit_posts`

```sql
CREATE TABLE reddit_posts (
    -- Primary key
    post_id TEXT PRIMARY KEY,

    -- Reddit metadata
    subreddit TEXT NOT NULL,
    title TEXT NOT NULL,
    selftext TEXT,
    author TEXT,
    created_utc TIMESTAMPTZ NOT NULL,
    score INTEGER,
    num_comments INTEGER,
    url TEXT,
    permalink TEXT,
    collected_at TIMESTAMPTZ NOT NULL,

    -- Sentiment analysis (VADER)
    sentiment_pos REAL,
    sentiment_neg REAL,
    sentiment_neu REAL,
    sentiment_compound REAL,
    sentiment_label TEXT,

    -- Vector embeddings
    embedding vector(384)
);
```

### Indexes

```sql
-- Metadata indexes
CREATE INDEX idx_subreddit ON reddit_posts(subreddit);
CREATE INDEX idx_created_utc ON reddit_posts(created_utc DESC);
CREATE INDEX idx_sentiment_label ON reddit_posts(sentiment_label);
CREATE INDEX idx_sentiment_compound ON reddit_posts(sentiment_compound);

-- Vector similarity indexes
CREATE INDEX idx_embedding_cosine ON reddit_posts
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX idx_embedding_l2 ON reddit_posts
USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);
```

### Functions

```sql
-- Semantic search function
CREATE OR REPLACE FUNCTION search_similar_posts(
    query_embedding vector(384),
    match_threshold float,
    match_count int,
    filter_subreddit text DEFAULT NULL,
    filter_sentiment text DEFAULT NULL,
    days_ago int DEFAULT 365
)
RETURNS TABLE (
    post_id text,
    subreddit text,
    title text,
    selftext text,
    sentiment_label text,
    sentiment_compound real,
    score integer,
    num_comments integer,
    permalink text,
    created_utc timestamptz,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.post_id,
        p.subreddit,
        p.title,
        p.selftext,
        p.sentiment_label,
        p.sentiment_compound,
        p.score,
        p.num_comments,
        p.permalink,
        p.created_utc,
        1 - (p.embedding <=> query_embedding) as similarity
    FROM reddit_posts p
    WHERE
        (filter_subreddit IS NULL OR p.subreddit = filter_subreddit)
        AND (filter_sentiment IS NULL OR p.sentiment_label = filter_sentiment)
        AND p.created_utc >= NOW() - INTERVAL '1 day' * days_ago
        AND p.embedding IS NOT NULL
        AND 1 - (p.embedding <=> query_embedding) > match_threshold
    ORDER BY p.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Database statistics function
CREATE OR REPLACE FUNCTION get_database_stats()
RETURNS TABLE (
    total_posts bigint,
    posts_with_embeddings bigint,
    posts_with_sentiment bigint,
    subreddits_count bigint,
    date_range text
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::bigint as total_posts,
        COUNT(embedding)::bigint as posts_with_embeddings,
        COUNT(sentiment_label)::bigint as posts_with_sentiment,
        COUNT(DISTINCT subreddit)::bigint as subreddits_count,
        (MIN(created_utc)::text || ' to ' || MAX(created_utc)::text) as date_range
    FROM reddit_posts;
END;
$$;
```

---

## 📊 Data Flow Diagrams

### Query Processing Flow

```
┌──────────────────────────────────────────────────────────┐
│  User Input: "What do people think about iPhone 15?"     │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│  embedder.embed_query()                                   │
│  - Load cached model (SentenceTransformer)               │
│  - Encode text → 384-dim vector                          │
│  - Time: ~0.1s                                           │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│  retriever.retrieve_similar_posts()                       │
│  - Call Supabase RPC: search_similar_posts()             │
│  - pgvector: cosine similarity search                    │
│  - Apply filters (subreddit, sentiment, date)            │
│  - Return top 15 posts                                   │
│  - Time: ~0.3s                                           │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│  prompt_templates.build_context_from_posts()             │
│  - Select top 10 posts                                   │
│  - Format with metadata                                  │
│  - Add sentiment distribution                            │
│  - Time: ~0.01s                                          │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│  groq_client.generate_completion()                       │
│  - Build system + user prompt                            │
│  - Call Groq API (llama-3.3-70b)                        │
│  - Stream/receive response                               │
│  - Time: ~2s                                             │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│  generator.validate_answer_quality()                     │
│  - Check citation presence                               │
│  - Validate response length                              │
│  - Extract cited sources                                 │
│  - Time: ~0.01s                                          │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│  Return Result Dictionary                                │
│  {                                                       │
│    'answer': "...",                                      │
│    'sources': [...],                                     │
│    'metadata': {                                         │
│      'timing': {...},                                    │
│      'citations': [...],                                 │
│      'posts_used': 10                                    │
│    }                                                     │
│  }                                                       │
└──────────────────────────────────────────────────────────┘

Total Time: ~2.5 seconds
```

---

## 🔐 Security Considerations

### API Key Management

```python
# ✅ GOOD: Use environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ❌ BAD: Hardcode in code
GROQ_API_KEY = "gsk_xxx..."  # Never do this!
```

### Database Access

```python
# Use service role key for server-side operations
# Use anon key for client-side operations (with RLS)
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
```

### Input Validation

```python
# Always validate user input
def validate_query(question: str) -> bool:
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    if len(question) > 1000:
        raise ValueError("Question too long (max 1000 chars)")
    return True
```

---

## ⚡ Performance Optimization

### Caching Strategy

**Model Caching:**
```python
@lru_cache(maxsize=1)  # Singleton pattern
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)
```

**Query Result Caching (Optional):**
```python
# Can add Redis caching for frequently asked questions
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_query(question: str) -> str:
    return pipeline.query(question)
```

### Database Optimization

**Index Tuning:**
```sql
-- ivfflat parameters
-- lists: number of clusters (good default: sqrt(row_count))
CREATE INDEX ON reddit_posts
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);  -- For ~10K-100K rows

-- For 1M+ rows, increase lists:
WITH (lists = 1000);
```

**Connection Pooling:**
```python
# Supabase client handles connection pooling internally
# For custom needs:
from supabase import create_client

client = create_client(
    supabase_url,
    supabase_key,
    options={
        'db': {
            'pool_size': 10  # Adjust based on load
        }
    }
)
```

### Batch Processing

```python
# Process multiple queries efficiently
questions = ["Question 1", "Question 2", "Question 3"]

# ❌ BAD: Load model each time
for q in questions:
    model = SentenceTransformer(...)
    result = query(q)

# ✅ GOOD: Load once, reuse
pipeline = RAGPipeline()
results = [pipeline.query(q) for q in questions]
```

---

## 🧪 Testing

### Unit Tests

```python
# Test embedder
def test_embed_query():
    embedding = embed_query("test query")
    assert len(embedding) == 384
    assert all(isinstance(x, float) for x in embedding)

# Test retriever
def test_retrieve_similar_posts():
    embedding = [0.1] * 384  # Dummy embedding
    posts = retrieve_similar_posts(embedding, top_k=5)
    assert len(posts) <= 5
    assert all('similarity' in post for post in posts)

# Test generator
def test_generate_answer():
    posts = [{'title': 'Test', 'selftext': 'Content'}]
    result = generate_answer("Test question", posts)
    assert 'answer' in result
    assert 'sources' in result
    assert 'metadata' in result
```

### Integration Tests

```python
# Test full pipeline
def test_full_pipeline():
    pipeline = RAGPipeline()
    result = pipeline.query("What do people think about iPhone?")

    assert result['answer']
    assert len(result['sources']) > 0
    assert result['metadata']['has_citations']
    assert result['metadata']['posts_used'] > 0
```

### Performance Tests

```python
import time

def test_query_performance():
    pipeline = RAGPipeline()

    start = time.time()
    result = pipeline.query("Test question")
    elapsed = time.time() - start

    assert elapsed < 5.0  # Should complete in < 5 seconds
```

---

## 📈 Monitoring and Logging

### Query Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='rag_queries.log'
)

logger = logging.getLogger('rag_pipeline')

# Log each query
logger.info(f"Query: {question}")
logger.info(f"Retrieved: {len(posts)} posts")
logger.info(f"Response time: {elapsed:.2f}s")
```

### Performance Metrics

```python
# Track metrics
metrics = {
    'total_queries': 0,
    'avg_response_time': 0,
    'cache_hits': 0,
    'errors': 0
}

# Example tracking
def track_query(result):
    metrics['total_queries'] += 1
    timing = result['metadata']['timing']
    metrics['avg_response_time'] = (
        (metrics['avg_response_time'] * (metrics['total_queries'] - 1) +
         timing['total_time']) / metrics['total_queries']
    )
```

---

## 🚀 Deployment Considerations

### Environment Variables

```bash
# Required
GROQ_API_KEY=gsk_...
SUPABASE_URL=https://...
SUPABASE_SERVICE_KEY=eyJh...

# Optional
VERBOSE=false
DEBUG_MODE=false
RESPONSE_STYLE=balanced
DEFAULT_TOP_K=15
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

### Resource Requirements

**Minimum:**
- CPU: 2 cores
- RAM: 2GB
- Storage: 1GB (for model)
- Network: Stable internet

**Recommended:**
- CPU: 4 cores
- RAM: 4GB
- Storage: 5GB
- Network: High bandwidth

---

## 🙏 Acknowledgments

**Technologies:**
- Sentence-Transformers by UKPLab
- Groq for ultra-fast LLM inference
- Supabase for vector database
- pgvector for PostgreSQL vector support
- Meta's Llama-3 model family

**Built as part of CSE299 (Junior Design Project)**
**North South University, Fall 2025**
