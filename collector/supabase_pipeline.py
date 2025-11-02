"""
Unified Supabase Pipeline for GitHub Actions
Orchestrates: Collection → Sentiment Analysis → Embedding Generation

This file is a pure orchestrator - all logic is imported from existing modules.
Zero duplication. Maximum maintainability.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from supabase_db.db_client import get_client
from reddit_config import get_reddit_client

# Import ALL existing functions - ZERO duplication
from collector.github_collector import (
    collect_from_subreddit,
    SUBREDDITS,
    FEED_LIMITS
)
from embeddings.generate_embeddings import EmbeddingGenerator
from analyzer.sentiment_utils import calculate_sentiment, prepare_text_for_sentiment

# Configuration
BATCH_SIZE = 100
GENERATE_EMBEDDINGS = os.getenv('GENERATE_EMBEDDINGS', 'false').lower() == 'true'


def enrich_posts_with_sentiment(
    posts: List[Dict[str, Any]], 
    analyzer: SentimentIntensityAnalyzer
) -> List[Dict[str, Any]]:
    """
    Add sentiment analysis to collected posts
    
    This is the ONLY custom logic in this file - it bridges collection and sentiment.
    
    Args:
        posts: List of post dictionaries from collector
        analyzer: VADER analyzer instance
    
    Returns:
        Posts with sentiment fields added
    """
    enriched_posts = []
    
    for post in posts:
        # Use shared utility to prepare text
        text = prepare_text_for_sentiment(
            title=post.get('title', ''),
            body=post.get('selftext', '')
        )
        
        # Use shared utility to calculate sentiment
        sentiment = calculate_sentiment(text, analyzer)
        
        # Merge sentiment into post
        enriched_posts.append({**post, **sentiment})
    
    return enriched_posts


def collect_all_posts(reddit) -> List[Dict[str, Any]]:
    """
    Collect posts from all configured subreddits
    Pure orchestration - delegates to existing collector
    
    Args:
        reddit: Reddit client instance
    
    Returns:
        List of collected posts (raw, no sentiment)
    """
    print("\n[COLLECTION] Gathering posts from subreddits...")
    all_posts = []
    
    for subreddit_name in SUBREDDITS:
        posts = collect_from_subreddit(reddit, subreddit_name)
        all_posts.extend(posts)
        time.sleep(2)  # Rate limiting
    
    print(f"[OK] Collected {len(all_posts):,} posts total")
    return all_posts


def insert_posts_to_supabase(supabase, posts: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Insert posts to Supabase in batches
    
    Args:
        supabase: Supabase client
        posts: List of posts to insert
    
    Returns:
        Dictionary with success/error counts
    """
    print(f"\n[INSERTION] Uploading {len(posts):,} posts to Supabase...")
    result = supabase.insert_posts(posts, batch_size=BATCH_SIZE)
    
    print(f"[OK] Inserted {result['success']:,} posts")
    if result['errors'] > 0:
        print(f"[INFO] {result['errors']:,} posts skipped (likely duplicates)")
    
    return result


def generate_embeddings_if_enabled(supabase):
    """
    Generate embeddings for posts without them (if enabled)
    Pure orchestration - delegates to existing generator
    
    Args:
        supabase: Supabase client (unused, kept for API consistency)
    """
    print(f"\n[EMBEDDINGS] Checking embedding status...")
    
    if not GENERATE_EMBEDDINGS:
        print("[SKIP] Embedding generation disabled")
        print("[INFO] Set GENERATE_EMBEDDINGS=true to enable")
        return
    
    try:
        print("[RUNNING] Generating embeddings for new posts...")
        generator = EmbeddingGenerator()
        generator.process_all_posts()
        print("[OK] Embedding generation complete")
        
    except Exception as e:
        print(f"[WARNING] Embedding generation failed: {e}")
        print("[INFO] Embeddings can be generated later")


def print_statistics(supabase, collected_posts: List[Dict[str, Any]]):
    """
    Print pipeline statistics and breakdowns
    
    Args:
        supabase: Supabase client
        collected_posts: Posts collected in this run
    """
    print("\n" + "="*60)
    print("PIPELINE STATISTICS")
    print("="*60)
    
    # Database statistics
    stats = supabase.get_stats()
    if stats:
        print(f"\n[DATABASE] Overall stats:")
        print(f"  Total posts:          {stats.get('total_posts', 0):,}")
        print(f"  Posts with sentiment: {stats.get('posts_with_sentiment', 0):,}")
        print(f"  Posts with embeddings:{stats.get('posts_with_embeddings', 0):,}")
        print(f"  Avg sentiment:        {stats.get('avg_sentiment_compound', 0):.3f}")
    
    # Breakdown by subreddit (this batch)
    subreddit_counts = Counter(post['subreddit'] for post in collected_posts)
    print(f"\n[THIS BATCH] Posts by subreddit:")
    for sub, count in subreddit_counts.most_common():
        print(f"  r/{sub:20s} {count:4d} posts")
    
    # Sentiment distribution (this batch)
    if collected_posts and 'sentiment_label' in collected_posts[0]:
        sentiment_counts = Counter(post['sentiment_label'] for post in collected_posts)
        print(f"\n[THIS BATCH] Sentiment distribution:")
        total = len(collected_posts)
        for label, count in sentiment_counts.items():
            percent = (count / total) * 100
            print(f"  {label.upper():10s}: {count:4d} posts ({percent:5.1f}%)")


def main():
    """
    Main pipeline orchestrator
    
    This is a pure orchestration function - it has NO business logic.
    All work is delegated to imported modules.
    
    Steps:
    1. Initialize clients (Reddit, Supabase, VADER)
    2. Collect posts (delegates to github_collector)
    3. Analyze sentiment (delegates to sentiment_utils)
    4. Insert to database (delegates to db_client)
    5. Generate embeddings (delegates to generate_embeddings)
    6. Report statistics
    """
    print("="*60)
    print("SUPABASE PIPELINE - AUTOMATED COLLECTION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    try:
        # [1] Initialize
        print("\n[1/5] Initializing clients...")
        reddit = get_reddit_client()
        supabase = get_client()
        analyzer = SentimentIntensityAnalyzer()
        print("[OK] Reddit, Supabase, and VADER ready")

        # [2] Collect
        print("\n[2/5] Collecting posts...")
        raw_posts = collect_all_posts(reddit)
        
        if not raw_posts:
            print("\n[WARNING] No posts collected! Exiting.")
            return

        # [3] Analyze
        print("\n[3/5] Analyzing sentiment...")
        posts_with_sentiment = enrich_posts_with_sentiment(raw_posts, analyzer)
        print(f"[OK] Analyzed sentiment for {len(posts_with_sentiment):,} posts")

        # [4] Insert
        print("\n[4/5] Inserting to database...")
        insert_posts_to_supabase(supabase, posts_with_sentiment)

        # [5] Embed
        print("\n[5/5] Processing embeddings...")
        generate_embeddings_if_enabled(supabase)

        # [6] Report
        print_statistics(supabase, posts_with_sentiment)

        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()