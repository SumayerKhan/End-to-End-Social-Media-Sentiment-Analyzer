"""
Generate vector embeddings for all posts in Supabase
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
import torch
from supabase_db.db_client import get_client
from embeddings.config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    EMBEDDING_BATCH_SIZE,
    UPDATE_BATCH_SIZE,
    MAX_TEXT_LENGTH,
    COMBINE_TITLE_BODY,
    DEVICE,
    SHOW_PROGRESS
)


class EmbeddingGenerator:
    """Generate embeddings for Reddit posts"""

    def __init__(self):
        """Initialize embedding generator"""
        print(f"Loading model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
        print(f"Model loaded! Embedding dimension: {EMBEDDING_DIMENSION}")

        self.supabase = get_client()

    def prepare_text(self, post: Dict[str, Any]) -> str:
        """
        Prepare text for embedding

        Args:
            post: Post dictionary

        Returns:
            Prepared text string
        """
        if COMBINE_TITLE_BODY:
            # Combine title and selftext
            title = post.get('title', '')
            selftext = post.get('selftext', '') or ''

            # Truncate if too long
            text = f"{title}\n{selftext}"
            if len(text) > MAX_TEXT_LENGTH:
                text = text[:MAX_TEXT_LENGTH]

            return text
        else:
            # Just use title
            return post.get('title', '')[:MAX_TEXT_LENGTH]

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=SHOW_PROGRESS,
            convert_to_numpy=True
        )

        # Convert to list of lists
        return embeddings.tolist()

    def process_all_posts(self):
        """Generate embeddings for all posts without embeddings"""
        print("\nFetching posts without embeddings...")
        posts = self.supabase.get_posts_without_embeddings()
        total_posts = len(posts)

        if total_posts == 0:
            print("[OK] All posts already have embeddings!")
            return

        print(f"Found {total_posts:,} posts to process\n")

        # Prepare texts
        print("Preparing texts...")
        texts = [self.prepare_text(post) for post in posts]

        # Generate embeddings
        print(f"Generating embeddings in batches of {EMBEDDING_BATCH_SIZE}...")
        start_time = time.time()

        embeddings = self.generate_embeddings(texts)

        embedding_time = time.time() - start_time
        posts_per_second = total_posts / embedding_time

        print(f"\n[OK] Generated {total_posts:,} embeddings in {embedding_time:.1f}s")
        print(f"[SPEED] ~{posts_per_second:.0f} posts/second")

        # Prepare updates
        print("\nPreparing updates...")
        updates = [
            {
                'post_id': posts[i]['post_id'],
                'embedding': embeddings[i]
            }
            for i in range(total_posts)
        ]

        # Upload to Supabase
        print(f"Uploading to Supabase in batches of {UPDATE_BATCH_SIZE}...")
        upload_start = time.time()

        total_batches = (total_posts + UPDATE_BATCH_SIZE - 1) // UPDATE_BATCH_SIZE
        success_count = 0

        for i in range(0, total_posts, UPDATE_BATCH_SIZE):
            batch = updates[i:i + UPDATE_BATCH_SIZE]
            batch_num = i // UPDATE_BATCH_SIZE + 1

            # Show progress
            progress = (i + len(batch)) / total_posts * 100
            bar_length = 40
            filled = int(bar_length * progress / 100)
            bar = '=' * filled + '-' * (bar_length - filled)

            print(f"\rBatch {batch_num}/{total_batches} [{bar}] {progress:.1f}%", end='', flush=True)

            try:
                result = self.supabase.update_embeddings(batch, batch_size=len(batch))
                success_count += result['success']
            except Exception as e:
                print(f"\nError in batch {batch_num}: {e}")

        print()  # New line after progress bar

        upload_time = time.time() - upload_start
        total_time = time.time() - start_time

        # Print summary
        print("\n" + "="*60)
        print("Embedding generation complete!")
        print("="*60)
        print(f"[OK] Processed: {success_count:,} posts")
        print(f"[TIME] Embedding time: {int(embedding_time//60)}m {int(embedding_time%60)}s")
        print(f"[TIME] Upload time: {int(upload_time//60)}m {int(upload_time%60)}s")
        print(f"[TIME] Total time: {int(total_time//60)}m {int(total_time%60)}s")
        print(f"[SPEED] Average: ~{success_count/total_time:.0f} posts/second")

    def generate_for_new_posts(self, post_ids: List[str] = None):
        """
        Generate embeddings for specific posts or recent posts

        Args:
            post_ids: Optional list of post IDs to process
        """
        if post_ids:
            print(f"Generating embeddings for {len(post_ids)} specific posts...")
            # TODO: Implement fetching specific posts
        else:
            # Process all posts without embeddings
            self.process_all_posts()


def main():
    """Main function"""
    try:
        generator = EmbeddingGenerator()
        generator.process_all_posts()

        # Show stats
        print("\n" + "="*60)
        print("Database Statistics:")
        print("="*60)
        stats = generator.supabase.get_stats()
        if stats:
            print(f"Total posts: {stats.get('total_posts', 0):,}")
            print(f"Posts with embeddings: {stats.get('posts_with_embeddings', 0):,}")
            print(f"Coverage: {stats.get('posts_with_embeddings', 0) / stats.get('total_posts', 1) * 100:.1f}%")

        print("\n[OK] Embedding generation completed successfully!")
        print("\n[NOTE] Next steps:")
        print("   1. Test search: python supabase_db/test_search.py")
        print("   2. Build RAG pipeline: rag/retriever.py")
        print("   3. Integrate LLM: rag/generator.py")

    except Exception as e:
        print(f"\n[ERROR] Embedding generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
