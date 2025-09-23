
from faker import Faker
import statistics
import time

fake = Faker()

# Generate your dataset
def generate_dataset(num_entries=1000, num_queries=100):
    dataset = []
    dataset_total_chars = 0
    queries = []
    queries_total_chars = 0
    
    for i in range(num_entries):
        # Generate text content
        text_content = fake.paragraph(nb_sentences=3)
        # Store the entry
        dataset.append({'text': text_content})
        # Track character counts
        char_count = len(text_content)
        dataset_total_chars += char_count

    # Generate realistic search queries
    for i in range(num_queries):
        query = fake.sentence(nb_words=5)  # 5-word sentences
        queries.append(query)
        char_count = len(query)
        queries_total_chars += char_count

    dataset_avg_chars =  dataset_total_chars / len(dataset)
    queries_avg_chars = queries_total_chars / len(queries)

    data = {
        "dataset": {
            "data": dataset, 
            "total chars": dataset_total_chars,
            "avg chars": dataset_avg_chars
        },
        "queries": {
            "data": queries, 
            "total chars": queries_total_chars,
            "avg chars": queries_avg_chars
        }
    }
    
    return data

# Generate
data = generate_dataset(1000)

# Print comprehensive dataset summary
print("=== DATASET SUMMARY ===")
print(f"Total dataset entries: {len(data['dataset']['data']):,}")
print(f"Total dataset characters: {data['dataset']['total chars']:,}")
print(f"Average dataset characters per entry: {data['dataset']['avg chars']:.1f}")

print(f"Total query entries: {len(data['queries']['data']):,}")
print(f"Total query characters: {data['queries']['total chars']:,}")
print(f"Average query characters per entry: {data['queries']['avg chars']:.1f}")