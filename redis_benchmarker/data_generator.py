"""Data generation utilities for Redis benchmarker testing."""

import random
import string
import time
from typing import List, Dict, Any, Optional
import click
import redis
import numpy as np
from faker import Faker
from tqdm import tqdm
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import math
import threading

fake = Faker()


class RedisDataGenerator:
    """Generate sample data for Redis benchmarking."""

    def __init__(self, redis_pool: redis.ConnectionPool):
        self.redis_pool = redis_pool
        self.fake = Faker()
        self._progress_lock = Lock()
        self._progress_counter = 0

    def generate_random_vector(self, dimension: int) -> List[float]:
        """Generate a random vector of specified dimension."""
        return np.random.random(dimension).astype(np.float32).tolist()

    def generate_product_document(self, doc_id: str, vector_dim: int) -> Dict[str, Any]:
        """Generate a sample product document with vector embedding."""
        categories = ["electronics", "clothing", "books", "home", "sports", "toys", "automotive"]
        brands = ["Apple", "Samsung", "Nike", "Adidas", "Sony", "Microsoft", "Amazon", "Google"]

        return {
            "id": doc_id,
            "title": self.fake.catch_phrase(),
            "description": self.fake.text(max_nb_chars=200),
            "category": random.choice(categories),
            "brand": random.choice(brands),
            "price": round(random.uniform(10.0, 1000.0), 2),
            "rating": round(random.uniform(1.0, 5.0), 1),
            "in_stock": random.choice([True, False]),
            "created_at": self.fake.date_time_between(start_date="-1y", end_date="now").isoformat(),
            "tags": random.sample(["sale", "new", "popular", "bestseller", "limited"], k=random.randint(0, 3)),
            "embedding": self.generate_random_vector(vector_dim)
        }

    def generate_user_document(self, doc_id: str, vector_dim: int) -> Dict[str, Any]:
        """Generate a sample user document with preference vector."""
        return {
            "id": doc_id,
            "name": self.fake.name(),
            "email": self.fake.email(),
            "age": random.randint(18, 80),
            "city": self.fake.city(),
            "country": self.fake.country(),
            "signup_date": self.fake.date_time_between(start_date="-2y", end_date="now").isoformat(),
            "preferences": self.generate_random_vector(vector_dim),
            "is_premium": random.choice([True, False])
        }

    def generate_article_document(self, doc_id: str, vector_dim: int) -> Dict[str, Any]:
        """Generate a sample article document with content vector."""
        topics = ["technology", "science", "politics", "sports", "entertainment", "health", "business"]

        return {
            "id": doc_id,
            "title": self.fake.sentence(nb_words=6),
            "content": self.fake.text(max_nb_chars=500),
            "author": self.fake.name(),
            "topic": random.choice(topics),
            "published_date": self.fake.date_time_between(start_date="-1y", end_date="now").isoformat(),
            "views": random.randint(100, 100000),
            "likes": random.randint(0, 1000),
            "content_embedding": self.generate_random_vector(vector_dim),
            "is_featured": random.choice([True, False])
        }

    def create_sample_index_schema(self, index_name: str, vector_field: str, vector_dim: int,
                                   document_type: str = "product") -> IndexSchema:
        """Create a sample index schema for different document types."""

        if document_type == "product":
            schema_dict = {
                "index": {
                    "name": index_name,
                    "prefix": f"{index_name}:product:",
                },
                "fields": [
                    {"name": "id", "type": "tag"},
                    {"name": "title", "type": "text"},
                    {"name": "description", "type": "text"},
                    {"name": "category", "type": "tag"},
                    {"name": "brand", "type": "tag"},
                    {"name": "price", "type": "numeric"},
                    {"name": "rating", "type": "numeric"},
                    {"name": "in_stock", "type": "tag"},
                    {"name": "created_at", "type": "text"},
                    {
                        "name": vector_field,
                        "type": "vector",
                        "attrs": {
                            "dims": vector_dim,
                            "distance_metric": "cosine",
                            "algorithm": "hnsw",
                            "datatype": "float32"
                        }
                    }
                ]
            }
        elif document_type == "user":
            schema_dict = {
                "index": {
                    "name": index_name,
                    "prefix": f"{index_name}:user:",
                },
                "fields": [
                    {"name": "id", "type": "tag"},
                    {"name": "name", "type": "text"},
                    {"name": "email", "type": "tag"},
                    {"name": "age", "type": "numeric"},
                    {"name": "city", "type": "text"},
                    {"name": "country", "type": "tag"},
                    {"name": "signup_date", "type": "text"},
                    {"name": "is_premium", "type": "tag"},
                    {
                        "name": "preferences",
                        "type": "vector",
                        "attrs": {
                            "dims": vector_dim,
                            "distance_metric": "cosine",
                            "algorithm": "hnsw",
                            "datatype": "float32"
                        }
                    }
                ]
            }
        elif document_type == "article":
            schema_dict = {
                "index": {
                    "name": index_name,
                    "prefix": f"{index_name}:article:",
                },
                "fields": [
                    {"name": "id", "type": "tag"},
                    {"name": "title", "type": "text"},
                    {"name": "content", "type": "text"},
                    {"name": "author", "type": "text"},
                    {"name": "topic", "type": "tag"},
                    {"name": "published_date", "type": "text"},
                    {"name": "views", "type": "numeric"},
                    {"name": "likes", "type": "numeric"},
                    {"name": "is_featured", "type": "tag"},
                    {
                        "name": "content_embedding",
                        "type": "vector",
                        "attrs": {
                            "dims": vector_dim,
                            "distance_metric": "cosine",
                            "algorithm": "hnsw",
                            "datatype": "float32"
                        }
                    }
                ]
            }
        else:
            raise ValueError(f"Unknown document type: {document_type}")

        return IndexSchema.from_dict(schema_dict)

    def insert_batch_documents(self, documents: List[Dict[str, Any]], key_prefix: str) -> int:
        """Insert a single batch of documents into Redis using pipeline."""
        redis_client = redis.Redis(connection_pool=self.redis_pool)
        pipe = redis_client.pipeline()

        for doc in documents:
            key = f"{key_prefix}:{doc['id']}"
            # Convert lists to comma-separated strings for Redis
            redis_doc = {}
            for k, v in doc.items():
                if isinstance(v, list) and k != "tags":  # Vector fields
                    redis_doc[k] = np.array(v, dtype=np.float32).tobytes()
                elif isinstance(v, list):  # Tag fields like tags
                    redis_doc[k] = ",".join(map(str, v))
                else:
                    redis_doc[k] = str(v)

            pipe.hset(key, mapping=redis_doc)

        pipe.execute()
        return len(documents)

    def generate_and_insert_worker_chunk(self, document_type: str, start_idx: int,
                                        end_idx: int, vector_dim: int, key_prefix: str,
                                        batch_size: int, pbar: tqdm) -> int:
        """Generate and insert a chunk of documents assigned to a worker thread."""
        generator_map = {
            "product": self.generate_product_document,
            "user": self.generate_user_document,
            "article": self.generate_article_document
        }

        if document_type not in generator_map:
            raise ValueError(f"Unknown document type: {document_type}")

        generator_func = generator_map[document_type]
        total_inserted = 0
        chunk_size = end_idx - start_idx

        for batch_start in range(start_idx, end_idx, batch_size):
            batch_end = min(batch_start + batch_size, end_idx)
            batch_documents = []

            # Generate one batch of documents
            for i in range(batch_start, batch_end):
                doc_id = f"{document_type}_{i:06d}"
                doc = generator_func(doc_id, vector_dim)
                batch_documents.append(doc)

            # Insert this batch and free memory
            inserted_batch = self.insert_batch_documents(batch_documents, key_prefix)
            total_inserted += inserted_batch

            # Update progress bar in a thread-safe manner
            with self._progress_lock:
                pbar.update(len(batch_documents))

            # Clear the batch from memory
            batch_documents.clear()

        return total_inserted

    def generate_and_insert_data(self, document_type: str, count: int, vector_dim: int,
                                key_prefix: str, batch_size: int = 100, num_workers: int = 1) -> int:
        """Generate and insert sample data using multiple worker threads."""
        click.echo(f"Generating {count} {document_type} documents using {num_workers} workers...")

        total_inserted = 0

        with tqdm(total=count, desc="Generating and inserting documents") as pbar:
            if num_workers == 1:
                # Single-threaded execution
                total_inserted = self.generate_and_insert_worker_chunk(
                    document_type, 0, count, vector_dim, key_prefix, batch_size, pbar
                )
            else:
                # Multi-threaded execution
                chunk_size = math.ceil(count / num_workers)

                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = []

                    for worker_id in range(num_workers):
                        start_idx = worker_id * chunk_size
                        end_idx = min(start_idx + chunk_size, count)

                        if start_idx < count:
                            future = executor.submit(
                                self.generate_and_insert_worker_chunk,
                                document_type, start_idx, end_idx, vector_dim,
                                key_prefix, batch_size, pbar
                            )
                            futures.append(future)

                    # Wait for all workers to complete and sum results
                    for future in as_completed(futures):
                        total_inserted += future.result()

        click.echo(f"Successfully inserted {total_inserted} documents")
        return total_inserted

    def generate_sample_document(self, document_type: str, vector_dim: int) -> Dict[str, Any]:
        """Generate a single sample document for display purposes."""
        generator_map = {
            "product": self.generate_product_document,
            "user": self.generate_user_document,
            "article": self.generate_article_document
        }

        if document_type not in generator_map:
            raise ValueError(f"Unknown document type: {document_type}")

        generator_func = generator_map[document_type]
        doc_id = f"{document_type}_sample"
        return generator_func(doc_id, vector_dim)


@click.command()
@click.option('--host', default='localhost', help='Redis host')
@click.option('--port', default=6379, help='Redis port')
@click.option('--password', default=None, help='Redis password')
@click.option('--db', default=0, help='Redis database number')
@click.option('--documents', default=10000, help='Number of documents to generate')
@click.option('--document-type', default='product',
              type=click.Choice(['product', 'user', 'article']),
              help='Type of documents to generate')
@click.option('--vector-dim', default=1536, help='Vector dimension')
@click.option('--index-name', default='sample_index', help='Name of the search index')
@click.option('--vector-field', default='embedding', help='Name of the vector field')
@click.option('--batch-size', default=100, help='Batch size for insertions')
@click.option('--create-index', is_flag=True, help='Create the search index')
@click.option('--force', is_flag=True, help='Force overwrite existing index')
@click.option('--workers', default=4, help='Number of worker threads')
def main(host, port, password, db, documents, document_type, vector_dim,
         index_name, vector_field, batch_size, create_index, force, workers):
    """Generate sample data for Redis benchmarking."""

    # Connect to Redis
    redis_pool = redis.ConnectionPool(
        host=host,
        port=port,
        password=password,
        db=db,
        decode_responses=False
    )

    try:
        redis_client = redis.Redis(connection_pool=redis_pool)
        redis_client.ping()
        click.echo(f"Connected to Redis at {host}:{port}")
    except Exception as e:
        click.echo(f"Failed to connect to Redis: {e}", err=True)
        return

    generator = RedisDataGenerator(redis_pool)

    # Adjust vector field name based on document type
    if document_type == "user":
        vector_field = "preferences"
    elif document_type == "article":
        vector_field = "content_embedding"

    # Create index if requested
    if create_index:
        try:
            redis_client = redis.Redis(connection_pool=redis_pool)
            # Check if index exists
            existing_indexes = [idx.decode() for idx in redis_client.execute_command("FT._LIST")]
            if index_name in existing_indexes:
                if force:
                    click.echo(f"Dropping existing index: {index_name}")
                    redis_client.execute_command("FT.DROPINDEX", index_name)
                else:
                    click.echo(f"Index {index_name} already exists. Use --force to overwrite.")
                    return

            click.echo(f"Creating search index: {index_name}")
            schema = generator.create_sample_index_schema(
                index_name, vector_field, vector_dim, document_type
            )

            search_index = SearchIndex(schema, redis_client)
            search_index.create()
            click.echo(f"Created index: {index_name}")

        except Exception as e:
            click.echo(f"Failed to create index: {e}", err=True)
            return

    # Generate key prefix
    key_prefix = f"{index_name}:{document_type}"

    # Generate and insert data
    start_time = time.time()
    inserted = generator.generate_and_insert_data(
        document_type, documents, vector_dim, key_prefix, batch_size, workers
    )
    end_time = time.time()

    click.echo(f"\nData generation completed in {end_time - start_time:.2f} seconds")
    click.echo(f"Generated {inserted} {document_type} documents")
    click.echo(f"Vector dimension: {vector_dim}")
    click.echo(f"Index name: {index_name}")
    click.echo(f"Vector field: {vector_field}")
    click.echo(f"Workers used: {workers}")

    # Show sample document
    if inserted > 0:
        click.echo("\nSample document structure:")
        sample = {k: v if not isinstance(v, list) or len(v) < 10 else f"[{len(v)} items]"
                 for k, v in generator.generate_sample_document(document_type, vector_dim).items()}
        for key, value in sample.items():
            click.echo(f"  {key}: {value}")


if __name__ == '__main__':
    main()