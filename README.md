![Data Augmenter Banner](https://github.com/marcoscobo/DataAugmenter/blob/main/docs/media/slim_banner.png?raw=true)

---

# Data Augmenter

[![License MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/marcoscobo/DataAugmenter/blob/main/LICENSE) [![Supported Python Versions](https://img.shields.io/pypi/pyversions/python-data-augmenter.svg)](https://pypi.org/project/python-data-augmenter/) [![PyPI Latest Release](https://img.shields.io/pypi/v/python-data-augmenter.svg)](https://pypi.org/project/python-data-augmenter/) [![Last commit](https://img.shields.io/github/last-commit/marcoscobo/DataAugmenter.svg)](https://github.com/marcoscobo/DataAugmenter/commits/main) [![Contributors](https://img.shields.io/github/contributors/marcoscobo/DataAugmenter.svg)](https://github.com/marcoscobo/DataAugmenter/graphs/contributors) [![Issues](https://img.shields.io/github/issues/marcoscobo/DataAugmenter.svg)](https://github.com/marcoscobo/DataAugmenter/issues)

Data Augmenter has been created to take advantage of the potential of foundational models by allowing us to generate new data from a small sample. Thanks to Data Augmenter we will be able to increase the size of our datasets by including variability in the data. In addition, we can extract structured datasets ready for fine-tuning of unstructured information.

## Installation

It is recommended to use conda environments to manage and install dependencies, but if you prefer to ignore it, skip directly to point 3.

1. Create an environment
   You can create a new environment using the conda create command. Replace myenv with your desired environment name and specify the Python version if needed.

   ```bash
   conda create --name myenv python=3.11
   ```
2. Activate the environment
   After creating the environment, activate it using the following command:

   ```bash
   conda activate myenv  
   ```

   You should now be working with the activated environment.
3. Installing dependencies
   Install ir from [PyPI](https://pypi.python.org/pypi/python-data-augmenter/) directly using pip:

   ```
   pip install python-data-augmenter
   ```

   At this point Data Augmenter is ready to use.

## Modules

This library consists of two modules, `augmentation` and `document_chunker`.

### Document Chunker

This module contains the `DocumentChunker` class. This utility has been designed to load and process specific types of files (`markdown`, `txt`, `pdf` and `jsonl`) by chunking them and inserting them in a dataframe.

#### Usage

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

### Augmentation

This module consists of two main types of classes: `Augmenters` and `Datasets`. `Augmenters` interface with Large Language Models (LLMs) through specified endpoints, providing the functionality to generate new data based on input documents. `Datasets` handle the dataset structure and offer methods for augmenting, filtering, and storing query-answer pairs relevant to the provided document.

The input dataset should be in the form of a DataFrame with a single column named "document" that contains chunks of your source document. The output will be a `.jsonl` file, where each entry includes a generated question-answer pair along with the corresponding document chunk. If filtering is applied, each entry will also include the cosine similarity score between the QA pair and its source chunk.

#### Usage

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
