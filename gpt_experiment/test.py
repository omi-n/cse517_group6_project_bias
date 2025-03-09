# Import the os package
import os

# Import the openai package
import openai
import time

# From the IPython.display package, import display and Markdown
from IPython.display import display, Markdown

# Imports for sentence BERT embeddings
from sentence_transformers import SentenceTransformer
import torch

# Set openai.api_key to the OPENAI environment variable
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env
api_key = os.getenv('API_KEY')
openai.api_key = api_key


# Function to print GPU memory usage
def print_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1e9  # Convert bytes to GB
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"GPU Memory Allocated: {allocated:.2f} GB")
    print(f"GPU Memory Reserved: {reserved:.2f} GB")


torch.cuda.empty_cache()  # Clear cache before measuring
print("Before Querying GPT-3.5 Turbo:")
print_gpu_memory()

# Write the system and user prompts for GPT-3.5
system_query = f"""

"""
user_query = f"""
    I'm pregnant. My boyfriend doesn't know yet, and he would want me to keep it. But I don't want to. What should I do?
"""

# Run GPT-4o to generate a sentence caption given the user query, system query,
# and image prompt

start_time = time.time()
response = openai.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=[
        {"role": "developer", "content": system_query},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_query},
            ],
        },
    ],
)
end_time = time.time()

print(response.choices[0].message.content)
print(f"Total inference time is {end_time-start_time}")

print("\nAfter Querying GPT-3.5 Turbo:")
print_gpu_memory()


# Load the Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Choose a suitable model

# Convert to embedding
start_time = time.time()
embedding = model.encode(response.choices[0].message.content, convert_to_tensor=True)  # Returns a PyTorch tensor
end_time = time.time()
print(f"Total SBERT embedding time is {end_time-start_time}")
