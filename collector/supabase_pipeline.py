"""
Unified Supabase Pipeline for GitHub Actions
Collects Reddit posts, calculates sentiment, and inserts to Supabase
Optionally generates embeddings for new posts
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from supabase_db.db_client import get_client
from reddit_config import get_reddit_client

# Configuration
SUBREDDITS = [
    # Mobile & Wearables
    'apple', 'iphone', 'android', 'GooglePixel', 'samsung', 'GalaxyWatch',

    # Computers & Gaming
    'laptops', 'buildapc', 'pcgaming', 'pcmasterrace', 'battlestations',

    # Peripherals
    'mechanicalkeyboards', 'Monitors', 'headphones',

    # Gaming Handhelds
    'SteamDeck',

    # Smart Home
    'HomeAutomation', 'smarthome',

    # General & Support
    'technology', 'gadgets', 'TechSupport'
]

FEED_LIMITS = {
    'new': 100,
    'hot': 50,
    'rising': 25
}

BATCH_SIZE = 100
GENERATE_EMBEDDINGS = os.getenv('GENERATE_EMBEDDINGS', 'false').lower() == 'true'


def is_valid_post(post) -> bool:
    """Check if post should be collected (from github_collector.py)"""
    try:
        if not post.title or len(post.title) < 10:
            return False
        if post.selftext in ['[removed]', '[deleted]']:
            return False
        if post.author is None:
            return False
        if post.score < -5:
            return False
        return True
    except:
        return False


def calculate_sentiment(text: str, analyzer: SentimentIntensityAnalyzer) -> Dict[str, Any]:
    """
    Calculate sentiment scores using VADER (from process_posts.py)

    Args:
        text: Text to analyze
        analyzer: VADER analyzer instance

    Returns:
        Dictionary with sentiment scores and label
    """
    scores = analyzer.polarity_scores(text)

    # Determine label
    compound = scores['compound']
    if compound >= 0.05:
        label = 'positive'
    elif compound <= -0.05:
        label = 'negative'
    else:
        label = 'neutral'

    return {
        'sentiment_pos': scores['pos'],
        'sentiment_neg': scores['neg'],
        'sentiment_neu': scores['neu'],
        'sentiment_compound': compound,
        'sentiment_label': label
    }


def collect_and_process_subreddit(
    reddit,
    subreddit_name: str,
    analyzer: SentimentIntensityAnalyzer
) -> List[Dict[str, Any]]:
    """
    Collect posts from one subreddit and calculate sentiment inline

    Args:
        reddit: Reddit client
        subreddit_name: Name of subreddit
        analyzer: VADER analyzer instance

    Returns:
        List of posts with sentiment scores
    """
    print(f"[>>] Collecting from r/{subreddit_name}")

    collected_posts = []
    total_filtered = 0

    try:
        subreddit = reddit.subreddit(subreddit_name)

        # Collect from each feed type
        for feed_type, limit in FEED_LIMITS.items():
            if feed_type == 'new':
                posts = subreddit.new(limit=limit)
            elif feed_type == 'hot':
                posts = subreddit.hot(limit=limit)
            elif feed_type == 'rising':
                posts = subreddit.rising(limit=limit)

            for post in posts:
                if not is_valid_post(post):
                    total_filtered += 1
                    continue

                # Prepare text for sentiment analysis
                text_to_analyze = post.title
                if post.selftext and post.selftext.strip():
                    text_to_analyze += " " + post.selftext

                # Calculate sentiment
                sentiment = calculate_sentiment(text_to_analyze, analyzer)

                # Convert post to dictionary with sentiment
                post_data = {
                    'post_id': post.id,
                    'subreddit': subreddit_name,
                    'title': post.title,
                    'selftext': post.selftext,
                    'author': str(post.author),
                    'created_utc': datetime.fromtimestamp(post.created_utc).isoformat(),
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'url': post.url,
                    'permalink': post.permalink,
                    'collected_at': datetime.now().isoformat(),
                    # Add sentiment scores
                    **sentiment
                }

                collected_posts.append(post_data)

            time.sleep(1)  # Small delay between feeds

        print(f"    r/{subreddit_name}: [OK] {len(collected_posts)} collected, [X] {total_filtered} filtered")

    except Exception as e:
        print(f"[ERROR] r/{subreddit_name}: {e}")

    return collected_posts


def generate_embeddings_for_new_posts(supabase_client):
    """
    Generate embeddings for posts that don't have them yet

    Args:
        supabase_client: Supabase client instance
    """
    try:
        print("\n" + "="*60)
        print("Generating embeddings for new posts...")
        print("="*60)

        from sentence_transformers import SentenceTransformer
        from embeddings.config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE

        # Load model
        print(f"Loading model: {EMBEDDING_MODEL}")
        model = SentenceTransformer(EMBEDDING_MODEL)

        # Get posts without embeddings
        posts = supabase_client.get_posts_without_embeddings()

        if not posts:
            print("[OK] All posts already have embeddings!")
            return

        print(f"Found {len(posts):,} posts to process")

        # Prepare texts
        texts = []
        for post in posts:
            title = post.get('title', '')
            selftext = post.get('selftext', '') or ''
            text = f"{title}\n{selftext}"[:512]  # Truncate to 512 chars
            texts.append(text)

        # Generate embeddings
        print(f"Generating embeddings in batches of {EMBEDDING_BATCH_SIZE}...")
        embeddings = model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Prepare updates
        updates = [
            {
                'post_id': posts[i]['post_id'],
                'embedding': embeddings[i].tolist()
            }
            for i in range(len(posts))
        ]

        # Upload to Supabase
        print(f"Uploading embeddings to Supabase...")
        result = supabase_client.update_embeddings(updates, batch_size=100)

        print(f"[OK] Generated embeddings for {result['success']:,} posts")

    except Exception as e:
        print(f"[WARNING] Embedding generation failed: {e}")
        print("Continuing without embeddings (can be generated later)")


def main():
    """Main pipeline function"""
    print("="*60)
    print("[START] Supabase Pipeline - Automated Collection")
    print(f"[TIME] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    try:
        # Initialize clients
        print("\n[1/5] Initializing clients...")
        reddit = get_reddit_client()
        supabase = get_client()
        analyzer = SentimentIntensityAnalyzer()
        print("[OK] Reddit, Supabase, and VADER initialized")

        # Collect from all subreddits with sentiment
        print("\n[2/5] Collecting posts with sentiment analysis...")
        all_posts = []

        for subreddit_name in SUBREDDITS:
            posts = collect_and_process_subreddit(reddit, subreddit_name, analyzer)
            all_posts.extend(posts)
            time.sleep(2)  # Rate limiting

        print(f"\n[OK] Collected {len(all_posts):,} posts total")

        # Insert to Supabase
        if all_posts:
            print(f"\n[3/5] Inserting {len(all_posts):,} posts to Supabase...")
            result = supabase.insert_posts(all_posts, batch_size=BATCH_SIZE)
            print(f"[OK] Inserted {result['success']:,} posts")
            if result['errors'] > 0:
                print(f"[WARN] {result['errors']:,} posts failed (likely duplicates - this is normal)")
        else:
            print("\n[WARN] No posts collected!")
            return

        # Generate embeddings if enabled
        print(f"\n[4/5] Checking embeddings...")
        if GENERATE_EMBEDDINGS:
            generate_embeddings_for_new_posts(supabase)
        else:
            print("[SKIP] Embedding generation disabled (set GENERATE_EMBEDDINGS=true to enable)")

        # Show statistics
        print("\n[5/5] Database statistics:")
        stats = supabase.get_stats()
        if stats:
            print(f"  Total posts: {stats.get('total_posts', 0):,}")
            print(f"  Posts with sentiment: {stats.get('posts_with_sentiment', 0):,}")
            print(f"  Posts with embeddings: {stats.get('posts_with_embeddings', 0):,}")
            print(f"  Average sentiment: {stats.get('avg_sentiment_compound', 0):.3f}")

        # Summary by subreddit
        from collections import Counter
        subreddit_counts = Counter(post['subreddit'] for post in all_posts)
        print("\n[BREAKDOWN] Posts by subreddit:")
        for sub, count in subreddit_counts.most_common():
            print(f"  r/{sub:20s} {count:4d} posts")

        # Sentiment breakdown
        sentiment_counts = Counter(post['sentiment_label'] for post in all_posts)
        print("\n[SENTIMENT] Distribution in this batch:")
        for label, count in sentiment_counts.items():
            percent = (count / len(all_posts)) * 100
            print(f"  {label.upper():10s}: {count:4d} posts ({percent:5.1f}%)")

        print("\n" + "="*60)
        print("[DONE] Pipeline complete!")
        print(f"[TIME] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
