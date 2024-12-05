import time

def process_batch(batch):
    """
    Simulating process time for each batch
    """
    print(f"Processing batch {batch}")
    time.sleep(1)

def batch(data, batch_size):
    """
    Processes data in batches.
    
    :param data: The dataset to be processed (list of items).
    :param batch_size: Number of items to process in each batch.
    """
    for i in range(0, len(data), batch_size):
        batch = data[i: i + batch_size]
        process_batch(batch)



# Sample data
data = [i for i in range(1, 21)]  # A list of numbers from 1 to 20

# Define batch size
batch_size = 5

# Process data in batches
batch(data, batch_size)

