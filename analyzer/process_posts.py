# analyzer/process_posts.py

import sqlite3
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime

# Database path
DB_PATH = os.path.join('database', 'tech_sentiment.db')

def process_posts_with_vader(limit=None):
    """
    Process posts with VADER sentiment analysis
    
    Args:
        limit: Number of posts to process (None = all posts)
    """
    
    print("=" * 70)
    print("VADER SENTIMENT ANALYSIS - PROCESSING POSTS")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize VADER
    analyzer = SentimentIntensityAnalyzer()
    
    try:
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get posts that haven't been processed yet
        query = """
            SELECT post_id, title, selftext 
            FROM raw_posts 
            WHERE sentiment_compound IS NULL
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        posts = cursor.fetchall()
        
        total_posts = len(posts)
        
        if total_posts == 0:
            print("\nNo posts to process (all already have sentiment scores)")
            conn.close()
            return
        
        print(f"\nFound {total_posts} posts to process")
        print("=" * 70)
        
        processed = 0
        skipped = 0
        
        # Process each post
        for post_id, title, selftext in posts:
            processed += 1
            
            # Combine title and text for analysis
            # Use title if selftext is empty
            text_to_analyze = ""
            
            if title:
                text_to_analyze += title
            
            if selftext and selftext.strip():
                if text_to_analyze:
                    text_to_analyze += " " + selftext
                else:
                    text_to_analyze = selftext
            
            # Skip if no text at all
            if not text_to_analyze.strip():
                skipped += 1
                continue
            
            # Get sentiment scores
            scores = analyzer.polarity_scores(text_to_analyze)
            
            # Determine label
            compound = scores['compound']
            if compound >= 0.05:
                label = 'positive'
            elif compound <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'
            
            # Update database
            cursor.execute("""
                UPDATE raw_posts 
                SET sentiment_pos = ?,
                    sentiment_neg = ?,
                    sentiment_neu = ?,
                    sentiment_compound = ?,
                    sentiment_label = ?
                WHERE post_id = ?
            """, (scores['pos'], scores['neg'], scores['neu'], 
                  compound, label, post_id))
            
            # Progress indicator every 100 posts
            if processed % 100 == 0:
                conn.commit()  # Save progress
                percent = (processed / total_posts) * 100
                print(f"Progress: {processed}/{total_posts} ({percent:.1f}%) - "
                      f"Last: {label.upper()} ({compound:.3f})")
        
        # Final commit
        conn.commit()
        
        # Get summary statistics
        cursor.execute("""
            SELECT 
                sentiment_label,
                COUNT(*) as count
            FROM raw_posts
            WHERE sentiment_label IS NOT NULL
            GROUP BY sentiment_label
        """)
        
        results = cursor.fetchall()
        
        print("\n" + "=" * 70)
        print("PROCESSING COMPLETE!")
        print("=" * 70)
        print(f"\nProcessed: {processed - skipped} posts")
        print(f"Skipped (no text): {skipped} posts")
        print(f"\nSENTIMENT BREAKDOWN:")
        print("-" * 40)
        
        total_analyzed = 0
        for label, count in results:
            total_analyzed += count
            print(f"  {label.upper():10s}: {count:6,d} posts")
        
        print("-" * 40)
        print(f"  {'TOTAL':10s}: {total_analyzed:6,d} posts")
        
        # Calculate percentages
        print(f"\nPERCENTAGES:")
        print("-" * 40)
        for label, count in results:
            percent = (count / total_analyzed) * 100
            print(f"  {label.upper():10s}: {percent:5.1f}%")
        
        print("\n" + "=" * 70)
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        conn.close()
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    # Process all posts
    # Change to process_posts_with_vader(1000) to test with 1000 posts first
    print("\nStarting sentiment analysis...")
    print("This may take a few minutes for 17,000+ posts...\n")
    
    success = process_posts_with_vader()
    
    if success:
        print("\nSUCCESS! All posts have been analyzed.")
        print("You can now check the database to see sentiment scores!")
    else:
        print("\nProcessing failed. Check errors above.")