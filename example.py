from augmenter.document_chunker import DocumentChunker
from augmenter.augmentation import DatasetAugmenter, OpenAIAugmenter
import os

# chunk_size: size of the chunk to be processed
# chunk_overlap: size of the overlap between chunks
chunker = DocumentChunker(chunk_size=2000, chunk_overlap=200)
dataset = chunker.process_file('data/cyberSEC.pdf')

custom_openai_system_prompt = """
You are a helpful cybersecurity assistant that only replies in JSONL
"""
custom_openai_user_prompt = """
You will be given a chunk of a cybersecurity. Your task is to create  {n} JSONL objects with two keys: "Question" and "Answer". The "Question" should ask how to perform the task described in the "Name" field. The "Answer" should provide the full instructions from the "Method" field.

Example:

Input:
{"Name": "open fault monitoring toolbox", "Method": "there are three ways to open the fault monitoring toolbox:\nfrom start menu\nfrom my apps\nfrom desktop icon\nthe fault monitoring toolbox window is displayed, \n"}

Expected Output:
{"Question": "What are the ways to open the fault monitoring toolbox?", "Answer": "There are three ways to open the fault monitoring toolbox:\nFrom start menu\nFrom my apps\nFrom desktop icon."}
{"Question": "How can you access the fault monitoring toolbox?", "Answer": "You can open the fault monitoring toolbox in three ways:\nVia the start menu\nThrough my apps\nUsing the desktop icon."}


Now, process the following entry and generate {n} question-answer pairs:

{document}

The information about the task is:

{document}"
"""

custom_openai_prompt = [
    {
        "role": "system",
        "content": custom_openai_system_prompt
    },
    {
        "role": "user",
        "content": custom_openai_user_prompt
    }
]

params = {
    "model": 'gpt-4o-mini',
    "messages": None,
    "frequency_penalty": 0,
    "logit_bias": {},
    "logprobs": False,
    "top_logprobs": None,
    "max_completion_tokens": None,
    "n": 1,
    "presence_penalty": 0,
    "response_format": None,
    "service_tier": None,
    "stop": None,
    "stream": False,
    "stream_options": None,
    "temperature": 1,
    "top_p": 1,
    "tool_choice": None,
    "user": None,
    "stream": False,
    "messages": custom_openai_prompt
}
augmenter = OpenAIAugmenter(params=params, api_key=os.getenv("OPENAI_API_KEY"))

print("Augmenting dataset...")

dataset_augmenter = DatasetAugmenter(augmenter=augmenter, dataset=dataset)

# n: number of augmentations per chunk
# m: number of times to replay the augmentation
# k: steps to save the augmented dataset to recovery from errors
dataset_augmenter.split_and_augment(n=50, m=3, k=20, max_threads=10)

print("Augmentation complete.")
print("Getting embeddings...")
dataset_augmenter.get_embeddings()

print("Getting cosine similarity...")
dataset_augmenter.get_cosine_similarity()
print("Getting cross cosine similarity...")
dataset_augmenter.get_cross_cosine_similarity()

# cosine_similarity_threshold: get rows with cosine similarity greater than this threshold
# cross_cosine_similarity_threshold: discard one of the two rows with cross cosine similarity greater than this threshold
dataset_augmenter.filter_dataset(cosine_similarity_threshold=0.5, cross_cosine_similarity_threshold=0.95)


print(f'Original dataset length: {len(dataset_augmenter.dataset)}')
print(f'Full augmented dataset length: {len(dataset_augmenter.augmented_dataset)}')
print(f'Filteres augmented dataset length: {len(dataset_augmenter.filtered_dataset)}')