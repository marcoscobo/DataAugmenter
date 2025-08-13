# Data Augmenter — API Reference

## Package structure

The diagram below sketches the top-level layout of the library.

* augmentation contains everything related to talking with LLM endpoints (augmenter.py) and orchestrating dataset generation (dataset.py).
* document_chunker focuses on slicing raw source files into uniform chunks via DocumentChunker.

Each sub-package ships its own __init__.py to expose a clean public API and to keep internal details encapsulated.

data_augmenter/
├── augmentation/
│           ├── __init__.py
│           ├── augmenter.py
│           └── dataset.py
└── document_chunker/
            ├── __init__.py
            └── document_chunker.py

The following sections describe the classes included in each module.

---

## Module `document_chunker`

### `class DocumentChunker`

| Type                | Name                                                  | Description                                                  |
| ------------------- | ----------------------------------------------------- | ------------------------------------------------------------ |
| **attribute** | `chunk_size: int \| None`                            | Maximum number of characters per chunk.                      |
| **attribute** | `chunk_overlap: int \| None`                         | Number of characters overlapping between consecutive chunks. |
| **attribute** | `separator: str \| None`                             | String used to split the text.                               |
| **method**    | `load_file(file_path: str) -> str`                  | Loads a file (`.md`, `.txt`, `.pdf`, `.jsonl`).      |
| **method**    | `chunk_document(doc: str) -> list[str]`             | Splits the document according to the defined parameters.     |
| **method**    | `save_to_dataframe() -> pd.DataFrame`               | Returns a DataFrame containing the generated chunks.         |
| **method**    | `process_and_chunk(file_path: str) -> pd.DataFrame` | Shortcut that combines loading and chunking.                 |

---

## Module `augmentation`

### `class TGIAugmenter`

| Type                | Name                                                                           | Description                                    |
| ------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------- |
| **attribute** | `endpoint: str`                                                              | TGI endpoint URL.                              |
| **attribute** | `params: dict \| None`                                                        | Optional inference parameters.                 |
| **attribute** | `prompt: str \| None`                                                         | Prompt template with dynamic placeholders.     |
| **method**    | `generate_queries_from_document(document: str, m: int, n: int) -> list[str]` | Generates `n` queries over `m` iterations. |

### `class OllamaAugmenter`

| Type                | Name                                                                           | Description               |
| ------------------- | ------------------------------------------------------------------------------ | ------------------------- |
| **attribute** | `endpoint: str`                                                              | Ollama endpoint URL.      |
| **attribute** | `model: str = 'llama3'`                                                      | Model to use.             |
| **attribute** | `prompt: str \| None`                                                         | Prompt template.          |
| **attribute** | `options: dict \| None`                                                       | Generation options.       |
| **method**    | `generate_queries_from_document(document: str, m: int, n: int) -> list[str]` | Same as `TGIAugmenter`. |

### `class OpenAIAugmenter`

| Type                | Name                                                                           | Description                  |
| ------------------- | ------------------------------------------------------------------------------ | ---------------------------- |
| **attribute** | `api_key: str`                                                               | OpenAI API key.              |
| **attribute** | `params: dict \| None`                                                        | Parameters for the API call. |
| **method**    | `generate_queries_from_document(document: str, m: int, n: int) -> list[str]` | Same as `TGIAugmenter`.    |

### `class DatasetAugmenter`

| Type                | Name                                                                                                                            | Description                                                                |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **attribute** | `augmenter`                                                                                                                   | Instance of `TGIAugmenter`, `OllamaAugmenter`, or `OpenAIAugmenter`. |
| **attribute** | `dataset: pd.DataFrame`                                                                                                       | Input DataFrame containing the `document` column.                        |
| **attribute** | `augmented_dataset`                                                                                                           | DataFrame resulting from the generation process.                           |
| **attribute** | `filtered_dataset`                                                                                                            | DataFrame after similarity filtering.                                      |
| **method**    | `split_and_augment(output_dir: str, output_file: str, m: int, n: int, k: int, max_threads: int, checkpoint_file: str \| None)` | Splits, generates QA pairs, and saves results with checkpoints.            |
| **method**    | `get_embeddings(output_dir: str, original_col: str, augmented_col: str, embeddings_model_id: str)`                            | Computes embeddings for originals and generated data.                      |
| **method**    | `get_cosine_similarity() -> pd.Series`                                                                                        | Cosine similarity between each QA pair and its original chunk.             |
| **method**    | `get_cross_cosine_similarity() -> np.ndarray`                                                                                 | Cross cosine similarity matrix between QA pairs.                           |
| **method**    | `filter_dataset(cosine_threshold: float, cross_threshold: float, output_file: str \| None)`                                    | Filters by similarity thresholds.                                          |
| **method**    | `load_augmented_dataset(file: str)`                                                                                           | Loads an already generated dataset.                                        |
| **method**    | `load_augmented_dataset_embeddings(orig_file: str, aug_file: str)`                                                            | Loads previously computed embeddings.                                      |
