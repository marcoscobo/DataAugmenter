# Data Augmenter

Data Augmenter has been created to take advantage of the potential of foundational models by allowing us to generate new data from a small sample. Thanks to Data Augmenter we will be able to increase the size of our datasets by including variability in the data. In addition, we can extract structured datasets ready for fine-tuning of unstructured information.

# Installing Prerequisites

It is recommended to use conda environments to manage and install dependencies, but if you prefer to ignore it, skip directly to point 3.

1. Create an environment

   You can create a new environment using the conda create command. Replace myenv with your desired environment name and specify the Python version if needed.

   ```bash
   conda create --name myenv python=3.10
   ```
2. Activate the environment

   After creating the environment, activate it using the following command:

   ```bash
   conda activate myenv  
   ```

   You should now be working with the activated environment.
3. Installing dependencies

   Acess the folder where Data Augmenter is downloaded to install the required dependencies with pip.

   ```
   cd ./path-to-installation
   pip install . --upgrade
   ```

   At this point Data Augmenter is ready to use.

# Modules

This library consists of two modules, "augmentation" and "document_chunker".

## Document Chunker

This module contains the `DocumentChunker` class. This utility has been designed to load and process specific types of files (`markdown`, `txt`, `pdf` and `jsonl`) by chunking them and inserting them in a dataframe.

### Class Details

#### `DocumentChunker`

* **Purpose** : Handles the loading, chunking, and structuring of documents in various formats.
* **Attributes** :
  * `chunk_size`: Optional maximum number of characters allowed in each chunk.
  * `chunk_overlap`: Optional number of characters that overlap between consecutive chunks, ensuring continuity.
  * `separator`: Optional character or string used to split the text into chunks.
* **Methods** :
  * `load_file(file_path)`: Loads the specified file and prepares it for chunking based on its format.
  * `chunk_document(document)`: Splits the loaded document into chunks according to the defined `chunk_size`, `chunk_overlap`, and `separator`, then stores the result in a DataFrame.
  * `save_to_dataframe()`: Saves the processed chunks into a DataFrame, ready for further augmentation or analysis.
  * `process_and_chunk(file_path)`: Combines file loading and chunking into a single method for streamlined processing.

### Usage

1. Initialize the DocumentChunker:

   ```python
   from document_chunker import DocumentChunker
   chunker = DocumentChunker(chunk_size, chunk_overlap, separator)
   ```
2. Process a File:

   ```python
   file_path = "path/to/your/file.txt"  # Can be .txt, .md, .pdf or .jsonl
   dataset = chunker.process_file(file_path)
   ```

   The output will be an augmentation-ready dataframe. In case you prefer to prepare your own dataset for augmentation, it should be a Pandas dataframe with a column named "document":

   ```python
   docs = [
       "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity.",
       "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world.",
       "All human beings should try to learn before they die what they are running from, and to, and why.",
       "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
       "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune, or to take arms against a sea of troubles and by opposing end them."
   ]
   dataset = pd.DataFrame({"document": docs})
   ```

## Augmentation

This module consists of two main types of classes: `Augmenters` and `Datasets`. `Augmenters` interface with Large Language Models (LLMs) through specified endpoints, providing the functionality to generate new data based on input documents. `Datasets` handle the dataset structure and offer methods for augmenting, filtering, and storing query-answer pairs relevant to the provided document.

The input dataset should be in the form of a DataFrame with a single column named "document" that contains chunks of your source document. The output will be a `.jsonl` file, where each entry includes a generated question-answer pair along with the corresponding document chunk. If filtering is applied, each entry will also include the cosine similarity score between the QA pair and its source chunk.

### Class Details

#### `TGIAugmenter`

* **Purpose**: Connects to a TGI (Text Generation Inference) endpoint to generate queries based on a provided document.
* **Attributes**:
  * `endpoint`: The URL of the TGI endpoint used to generate queries.
  * `params`: Optional dictionary of parameters to customize the LLM's behavior.
  * `prompt`: Optional template for the prompt used to generate queries, with placeholders for dynamic content.
* **Key Methods**:
  * `generate_queries_from_document(document, m, n)`: Generates `n` queries for `m` iterations using the specified LLM.

#### `OllamaAugmenter`

* **Purpose**: Connects to an Ollama endpoint to generate queries based on a provided document.
* **Attributes**:
  * `endpoint`: The URL of the Ollama endpoint used to generate queries.
  * `model`: Optional model name used for query generation, defaulting to `'llama3'`.
  * `prompt`: Optional template for the prompt used to generate queries, with placeholders for dynamic content.
  * `options`: Optional dictionary of settings to customize the query generation process.
* **Key Methods**:
  * `generate_queries_from_document(document, m, n)`: Generates `n` queries for `m` iterations using the specified LLM.

#### `DatasetAugmenter`

* **Purpose**: Manages the entire dataset augmentation process, from splitting to filtering.
* **Attributes**:
  * `augmenter`: An instance of either `TGIAugmenter` or `OllamaAugmenter` used to generate queries.
  * `dataset`: The input dataset, typically a DataFrame containing the document chunks.
  * `augmented_dataset`: Stores the augmented dataset after processing.
  * `filtered_dataset`: Stores the dataset after applying filtering methods.
  * `augmented_dataset_original_embeddings`: Stores embeddings of the original document chunks.
  * `augmented_dataset_augmented_embeddings`: Stores embeddings of the augmented data.
  * `cross_cosine_similarity_matrix`: Stores the cross cosine similarity matrix between the generated QA pairs.
* **Key Methods**:
  * `split_and_augment(output_dir, output_file, m, n, k, max_threads, checkpoint_file)`: Splits the dataset in `k` batches, generates QA pairs using `max_threads` threads (allowing query batching to the `Augmenter` class, increasing processing performance), each k batch in one thread, generating `n` queries for `m` iterations for each dataset row, and saves the results. The process stores checkpoint to recover the status in failure case.
  * `get_embeddings(output_dir, original_col, augmented_col, embeddings_model_id)`: Computes embeddings for the original and augmented data.
  * `get_cosine_similarity()`: Calculates cosine similarity for each QA pair and its original document.
  * `get_cross_cosine_similarity()`: Calculates a cross cosine similarity matrix between each possible pair of generated QA pairs.
  * `filter_dataset(cosine_similarity_threshold, cross_cosine_similarity_threshold, output_file)`: Filters the dataset based on cosine similarity thresholds.
  * `load_augmented_dataset(file)`: Load an augmented_dataset to the class from the specified file.
  * `load_augmented_dataset_embeddings(augmented_dataset_original_embeddings_file, augmented_dataset_augmented_embeddings_file)`: Load the augmented_dataset embeddings to the class from the specified files.

### Usage

For the following usage example, we have used a Ollama client exposed at localhost:11434 port 80 with the tinyllama 1.1b model.

1. Initialize TGIAugmenter:

   ```python
   from augmenter import TGIAugmenter
   augmenter = OllamaAugmenter("http://localhost:11434/api/generate", model='tinyllama:1.1b')
   ```
2. Initialize DatasetAugmenter:

   ```python
   from augmenter import DatasetAugmenter
   dataset_augmenter = DatasetAugmenter(augmenter=augmenter, dataset=dataset)
   ```
   After the process is finished, the dataset will be saved in the 'augmented_dataset.jsonl' file by default.
3. Generate the question and answer pairs:

   Optionally, filter the augmented dataset:

   ```python
   dataset_augmenter.filter_dataset(cosine_similarity_threshold=0.45, cross_cosine_similarity_threshold=0.85)
   ```
   This will automatically process the embeddings and filter the dataset based on the set thresholds.
   Alternatively it can be done manually:

   ```python
   dataset_augmenter.get_embeddings()
   dataset_augmenter.get_cosine_similarity()
   dataset_augmenter.get_cross_cosine_similarity()
   dataset_augmenter.filter_dataset(cosine_similarity_threshold=0.45, cross_cosine_similarity_threshold=0.85)
   ```
