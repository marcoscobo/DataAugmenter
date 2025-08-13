# [1.0.0] - 2025-01-01

## Added
- **Initial release of Data Augmenter** with two core modules:  
  - **`document_chunker`**
    - `DocumentChunker` class for loading `.md`, `.txt`, `.pdf` and `.jsonl` files, chunking text with configurable `chunk_size`, `chunk_overlap` and `separator`, and exporting the result to a `pandas.DataFrame`.
  - **`augmentation`**
    - `TGIAugmenter`, `OllamaAugmenter` and `OpenAIAugmenter` classes for generating queries via TGI, Ollama and OpenAI endpoints.
    - `DatasetAugmenter` class for end-to-end augmentation: dataset splitting, multi-threaded QA-pair generation, checkpointing, embedding calculation, cosine-similarity matrix creation, cross-similarity filtering and export to `.jsonl`.
- **Utilities**
  - Embedding helpers for original and augmented data.
  - Automatic filtering by cosine-similarity thresholds to keep only high-quality QA pairs.
- **Documentation**
  - Conda-based setup plus `pip install . --upgrade` workflow.
  - Usage examples for `DocumentChunker`, each `Augmenter`, and `DatasetAugmenter`.
  - Full API reference with attribute and method descriptions.

## Changed
*(none — first release)*

## Fixed
*(none — first release)*