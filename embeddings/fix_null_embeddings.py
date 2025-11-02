"""
Fix NULL embeddings in Supabase
This script generates embeddings for all posts that have NULL in the embedding column
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from embeddings.generate_embeddings import EmbeddingGenerator


def main():
    """Fix NULL embeddings in database"""
    print("="*60)
    print("FIX NULL EMBEDDINGS")
    print("="*60)
    print("\nThis will generate embeddings for all posts with NULL embeddings")
    print("This may take several minutes depending on the number of posts...\n")

    try:
        # Initialize generator
        generator = EmbeddingGenerator()

        # Process all posts without embeddings
        generator.process_all_posts()

        print("\n" + "="*60)
        print("[SUCCESS] All NULL embeddings have been fixed!")
        print("="*60)

    except Exception as e:
        print(f"\n[ERROR] Failed to fix embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
