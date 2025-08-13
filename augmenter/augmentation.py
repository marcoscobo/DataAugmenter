import pkg_resources
import requests
import pandas as pd
import numpy as np
import os
import json
import yaml
from tqdm import tqdm
import concurrent.futures
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from augmenter.utils import extract_jsonl_content, merge_jsonl_files
from openai import OpenAI
import copy


with open(pkg_resources.resource_filename('augmenter', 'config/default_tgi_params.yaml'), 'r') as file:
    default_tgi_params = yaml.safe_load(file)
with open(pkg_resources.resource_filename('augmenter', 'config/default_ollama_options.yaml'), 'r') as file:
    default_ollama_options = yaml.safe_load(file)
with open(pkg_resources.resource_filename('augmenter', 'config/default_tgi_prompt.txt'), 'r') as file:
    default_tgi_prompt = file.read()
with open(pkg_resources.resource_filename('augmenter', 'config/default_ollama_prompt.txt'), 'r') as file:
    default_ollama_prompt = file.read()


class TGIAugmenter:
    """
    Augmenter class that connects to a TGI (Text Generation Inference) endpoint 
    to generate synthetic queries for data augmentation.
    """
    def __init__(self, endpoint, params=None, prompt=None):
        """
        Initializes the TGI augmenter.

        Args:
            endpoint (str): The TGI endpoint URL.
            params (dict, optional): Generation parameters. If None, default config is used.
            prompt (str, optional): Prompt template string. If None, default is loaded.
        """
        self.endpoint = endpoint
        self.params = params or default_tgi_params
        self.prompt = prompt or default_tgi_prompt

    @staticmethod
    def call_tgi(endpoint, payload, params):
        """
        Sends a generation request to the TGI endpoint.

        Args:
            endpoint (str): TGI server URL.
            payload (str): The input text with prompt formatting.
            params (dict): Generation parameters for the LLM.

        Returns:
            str: The generated text from the model.
        """
        data = json.dumps({'inputs': payload, 'parameters': params})
        response = requests.post(endpoint, headers={'Content-Type': 'application/json'}, data=data)
        return response.json()[0]['generated_text']

    def generate_queries_from_document(self, document, m, n):
        """
        Runs `m` parallel generation tasks for a given document, each asking for `n` queries.

        Args:
            document (str): Input text to augment.
            m (int): Number of parallel generations.
            n (int): Number of queries to request in each generation.

        Returns:
            list: A list of parsed JSON objects (queries).
        """
        all_augmentations = []
        def query_task():
            try:
                payload = self.prompt.replace('{document}', document).replace('{n}', str(n))                
                response = self.call_tgi(endpoint=self.endpoint, payload=payload, params=self.params)
                return extract_jsonl_content(response)
            except Exception as e:
                print(f"Exception in query_task for document {document}: {repr(e)}")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(query_task) for _ in range(m)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    augmentations = future.result()
                    all_augmentations.extend(augmentations)
                except Exception as e:
                    print(f"Error occurred during query: {repr(e)}")
        return all_augmentations


class OllamaAugmenter:
    """
    Augmenter class that interfaces with an Ollama-compatible model endpoint for query generation.
    """
    def __init__(self, endpoint, model=None, prompt=None, options=None):
        """
        Initializes the Ollama augmenter.

        Args:
            endpoint (str): Ollama server URL.
            model (str, optional): Model ID (default: "llama3").
            prompt (str, optional): Prompt template with placeholders.
            options (dict, optional): Additional Ollama config options.
        """
        self.endpoint = endpoint
        self.model = model or 'llama3'
        self.prompt = prompt or default_ollama_prompt
        self.options = options or {}

    @staticmethod
    def call_ollama(endpoint, model, payload, options):
        """
        Sends a prompt to the Ollama API.

        Args:
            endpoint (str): Ollama endpoint URL.
            model (str): Model identifier.
            payload (str): Input prompt.
            options (dict): Generation options.

        Returns:
            str: Raw response text from the model.
        """
        data = json.dumps({'model': model, 'prompt': payload, 'options': options, 'stream': False})
        response = requests.post(endpoint, headers={'Content-Type': 'application/json'}, data=data)
        return response.json()['response']

    def generate_queries_from_document(self, document, m, n):
        """
        Executes multiple asynchronous query generations from a document.

        Args:
            document (str): Source document text.
            m (int): Number of times to invoke the model.
            n (int): Number of queries to generate per invocation.

        Returns:
            list: List of extracted JSON query objects.
        """
        all_augmentations = []
        def query_task():
            try:
                payload = self.prompt.replace('{document}', document).replace('{n}', str(n))                
                response = self.call_ollama(endpoint=self.endpoint, model=self.model, payload=payload, options=self.options)
                return extract_jsonl_content(response)
            except Exception as e:
                print(f"Exception in query_task for document {document}: {repr(e)}")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(query_task) for _ in range(m)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    augmentations = future.result()
                    all_augmentations.extend(augmentations)
                except Exception as e:
                    print(f"Error occurred during query: {repr(e)}")
        return all_augmentations

class OpenAIAugmenter:
    """
    Augmenter class that leverages OpenAI's Chat API to generate synthetic queries.
    """
    def __init__(self, params=None, api_key=None):
        """
        Initializes the OpenAI augmenter.

        Args:
            params (dict, optional): OpenAI chat completion parameters.
            api_key (str, optional): OpenAI API key.
        """
        self.params = params or default_tgi_params
        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def call_openAI(self, params):
        """
        Sends a completion request to the OpenAI API.

        Args:
            params (dict): Fully constructed chat completion request.

        Returns:
            str: Generated completion content.
        """
        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content

    def generate_queries_from_document(self, document, m, n):
        """
        Generates queries from document using OpenAI chat completions.

        Args:
            document (str): Text to use in prompt template.
            m (int): Number of generations.
            n (int): Number of queries per generation.

        Returns:
            list: Parsed JSON objects extracted from responses.
        """
        all_augmentations = []
        def query_task():
            try:
                params = copy.deepcopy(self.params)
                for message in params["messages"]:
                    if "content" in message and isinstance(message["content"], str):
                        message["content"] = message["content"].replace('{document}', document).replace('{n}', str(n))

                response = self.call_openAI(self=self, params=params)
                return extract_jsonl_content(response)
            except Exception as e:
                print(f"Exception in query_task for document {document}: {repr(e)}")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(query_task) for _ in range(m)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    augmentations = future.result()
                    all_augmentations.extend(augmentations)
                except Exception as e:
                    print(f"Error occurred during query: {repr(e)}")
        return all_augmentations

class DatasetAugmenter:
    """
    Orchestrates end-to-end dataset augmentation and filtering using an LLM-based augmenter.
    """
    def __init__(self, augmenter=None, dataset=None):
        """
        Initializes the augmenter with a model and dataset.

        Args:
            augmenter (object): An instance of a model augmenter class.
            dataset (DataFrame): Dataset to augment (must contain a 'document' column).
        """
        self.augmenter = augmenter
        self.dataset = dataset
        self.augmented_dataset = None
        self.filtered_dataset = None
        self.augmented_dataset_original_embeddings = None
        self.augmented_dataset_augmented_embeddings = None
        self.cross_cosine_similarity_matrix = None

    @staticmethod
    def augment_dataset(augmenter, dataset, m, n):
        """
        Applies the augmenter to all documents in a dataset.

        Args:
            augmenter (object): The model augmenter.
            dataset (DataFrame): The input dataset.
            m (int): Number of repetitions per document.
            n (int): Number of queries to generate per repetition.

        Returns:
            DataFrame: A dataset with original and augmented text pairs.
        """
        augmented_data = []
        for _, row in dataset.iterrows():
            document = row['document']
            augmentations = augmenter.generate_queries_from_document(document=document, m=m, n=n)
            for augmentation in augmentations:
                augmented_data.append({'original_document': document, 'augmented_data': augmentation})
        return pd.DataFrame(augmented_data)

    @staticmethod
    def save_checkpoint(array, checkpoint_file):
        """
        Saves checkpoint progress to disk.

        Args:
            array (list): List of integers marking completed splits.
            checkpoint_file (str): File path to save the checkpoint.
        """
        with open(checkpoint_file, 'w') as f:
            f.write(json.dumps(array))
    
    def process_split(self, index, split, output_dir, output_file, m, n):
        """
        Processes a single split of the dataset and saves output to disk.

        Args:
            index (int): Index of the current split.
            split (DataFrame): Subset of the dataset.
            output_dir (str): Directory to write the output.
            output_file (str): Base output filename.
            m (int): Repetitions per document.
            n (int): Queries per repetition.

        Returns:
            int: The index of the processed split.
        """
        augmented_df = self.augment_dataset(augmenter=self.augmenter, dataset=split, m=m, n=n)
        augmented_df.to_json(os.path.join(output_dir, output_file[:-6] + f'_part_{index}.jsonl'), orient='records', lines=True)
        return index

    def split_and_augment(self, output_dir='temp', output_file='augmented_dataset.jsonl', m=1, n=1, k=1, max_threads=1, checkpoint_file='augment_index.checkpoint'):
        """
        Splits the dataset into `k` parts, processes each split in parallel, and merges output.

        Args:
            output_dir (str): Temporary folder for intermediate files.
            output_file (str): Final merged output file.
            m (int): Repetitions per document.
            n (int): Queries per repetition.
            k (int): Number of splits.
            max_threads (int): Max number of concurrent threads.
            checkpoint_file (str): Path to the checkpoint tracking file.

        Raises:
            Exception: If augmenter or dataset is not initialized.
        """
        if self.augmenter is None or self.dataset is None:
            raise Exception('Augmenter or Dataset is empty')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint_array = json.loads(f.read().strip())
        else:
            checkpoint_array = [0] * k
        splits = np.array_split(self.dataset, k)
        completed_splits = checkpoint_array.count(1)
        with tqdm(total=len(splits), initial=completed_splits, desc="Processing Splits") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
                futures = {executor.submit(self.process_split, index, split, output_dir, output_file, m, n): index for index, split in enumerate(splits) if checkpoint_array[index] == 0}
                for future in concurrent.futures.as_completed(futures):
                    index = futures[future]
                    try:
                        future.result()
                        checkpoint_array[index] = 1
                        self.save_checkpoint(checkpoint_array, checkpoint_file)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing split {index}: {repr(e)}")
        merge_jsonl_files(folder_path=output_dir, output_file=output_file)
        self.augmented_dataset = pd.read_json(output_file, lines=True)
        self.augmented_dataset = self.augmented_dataset[self.augmented_dataset['augmented_data'] != {}]
    
    def get_embeddings(self, output_dir='temp', original_col='original_document', augmented_col='augmented_data', embeddings_model_id='paraphrase-MiniLM-L6-v2'):
        """
        Computes or loads sentence embeddings for both original and augmented text.

        Args:
            output_dir (str): Folder to store .npy files.
            original_col (str): Column with original text.
            augmented_col (str): Column with augmented text.
            embeddings_model_id (str): HuggingFace SentenceTransformer model ID.

        Raises:
            Exception: If augmented dataset is not available.
        """
        if self.augmented_dataset is None:
            raise Exception('Augmented Dataset is empty')
        embeddings_model = None
        original_embeddings_file = os.path.join(output_dir, 'augmented_dataset_original_embeddings.npy')
        augmented_embeddings_file = os.path.join(output_dir, 'augmented_dataset_augmented_embeddings.npy')
        with tqdm(total=2, initial=0, desc="Processing Embeddings") as pbar:
            if os.path.exists(original_embeddings_file):
                self.augmented_dataset_original_embeddings = np.load(original_embeddings_file)
                pbar.update(1)
            if os.path.exists(augmented_embeddings_file):
                self.augmented_dataset_augmented_embeddings = np.load(augmented_embeddings_file)
                pbar.update(1)
            if self.augmented_dataset_original_embeddings is None:
                if embeddings_model is None:
                    embeddings_model = SentenceTransformer(embeddings_model_id)
                self.augmented_dataset_original_embeddings = embeddings_model.encode(self.augmented_dataset[original_col].tolist(), show_progress_bar=False)
                np.save(original_embeddings_file, self.augmented_dataset_original_embeddings)
                pbar.update(1)
            if self.augmented_dataset_augmented_embeddings is None:
                if embeddings_model is None:
                    embeddings_model = SentenceTransformer(embeddings_model_id)
                self.augmented_dataset_augmented_embeddings = embeddings_model.encode(self.augmented_dataset[augmented_col].tolist(), show_progress_bar=False)
                np.save(augmented_embeddings_file, self.augmented_dataset_augmented_embeddings)
                pbar.update(1)

    def get_cosine_similarity(self):
        """
        Computes cosine similarity between original and augmented embeddings.

        Raises:
            Exception: If embeddings are missing.
        """
        if self.augmented_dataset_original_embeddings is None or self.augmented_dataset_augmented_embeddings is None:
            raise Exception('Embeddings has not been calculated')
        cosine_similarity_matrix = cosine_similarity(self.augmented_dataset_original_embeddings, self.augmented_dataset_augmented_embeddings)
        self.augmented_dataset.loc[:, 'cosine_similarity'] = cosine_similarity_matrix.diagonal()

    def get_cross_cosine_similarity(self):
        """
        Computes pairwise cosine similarity across all augmented embeddings.

        Raises:
            Exception: If embeddings are missing.
        """
        if self.augmented_dataset_augmented_embeddings is None:
            raise Exception('Embeddings has not been calculated')
        self.cross_cosine_similarity_matrix = cosine_similarity(self.augmented_dataset_augmented_embeddings)

    @staticmethod
    def filter_cosine_similarity(dataset, threshold):
        """
        Filters dataset rows with cosine similarity above threshold.

        Args:
            dataset (DataFrame): Augmented dataset with similarity column.
            threshold (float): Minimum similarity required.

        Returns:
            set: Indexes of documents to retain.
        """
        to_keep = set(dataset[abs(dataset['cosine_similarity']) > threshold].index)
        return to_keep
    
    @staticmethod
    def filter_cross_cosine_similarity(cross_cosine_similarity_matrix, threshold):
        """
        Filters out redundant augmented queries using cross-similarity matrix.

        Args:
            cross_cosine_similarity_matrix (ndarray): Matrix of pairwise similarities.
            threshold (float): Max allowed similarity between augmentations.

        Returns:
            set: Indexes of documents to retain.
        """
        to_keep = set(range(len(cross_cosine_similarity_matrix)))
        for i in range(len(cross_cosine_similarity_matrix)):
            for j in range(i + 1, len(cross_cosine_similarity_matrix)):
                if cross_cosine_similarity_matrix[i][j] > threshold:
                    if j in to_keep:
                        to_keep.discard(j)
        return to_keep

    def filter_dataset(self, cosine_similarity_threshold=None, cross_cosine_similarity_threshold=None, output_file='filtered_augmented_dataset.jsonl'):
        """
        Filters the dataset using cosine similarity and cross-similarity thresholds.

        Args:
            cosine_similarity_threshold (float, optional): Minimum semantic similarity.
            cross_cosine_similarity_threshold (float, optional): Maximum redundancy threshold.
            output_file (str): Path to save the filtered dataset.
        """
        if self.augmented_dataset is None:
            raise Exception('Augmented Dataset is empty')
        if self.augmented_dataset_original_embeddings is None or self.augmented_dataset_augmented_embeddings is None:
            self.get_embeddings()
        if cosine_similarity_threshold is not None and 'cosine_similarity' not in self.augmented_dataset.columns:
            self.get_cosine_similarity()
        if cross_cosine_similarity_threshold is not None and self.cross_cosine_similarity_matrix is None:
            self.get_cross_cosine_similarity()
        to_keep_cosine_similarity = self.filter_cosine_similarity(dataset=self.augmented_dataset, threshold=cosine_similarity_threshold) if cosine_similarity_threshold is not None else set(self.augmented_dataset.index)
        to_keep_cross_cosine_similarity = self.filter_cross_cosine_similarity(cross_cosine_similarity_matrix=self.cross_cosine_similarity_matrix, threshold=cross_cosine_similarity_threshold) if cross_cosine_similarity_threshold is not None else set(self.augmented_dataset.index)
        to_keep = list(to_keep_cosine_similarity & to_keep_cross_cosine_similarity)
        self.filtered_dataset = self.augmented_dataset.iloc[list(to_keep)]
        self.filtered_dataset.to_json(output_file, orient='records', lines=True)

    def load_augmented_dataset(self, file='augmented_dataset.jsonl'):
        """
        Loads an augmented dataset from disk.

        Args:
            file (str): Path to the JSONL dataset file.
        """
        self.augmented_dataset = pd.read_json(file, lines=True)
        self.augmented_dataset = self.augmented_dataset[self.augmented_dataset['augmented_data'] != {}]
    
    def load_augmented_dataset_embeddings(self, augmented_dataset_original_embeddings_file='temp/augmented_dataset_original_embeddings.npy', augmented_dataset_augmented_embeddings_file='temp/augmented_dataset_augmented_embeddings.npy'):
        """
        Loads embeddings from .npy files.

        Args:
            augmented_dataset_original_embeddings_file (str): Path to original embeddings file.
            augmented_dataset_augmented_embeddings_file (str): Path to augmented embeddings file.
        """
        self.augmented_dataset_original_embeddings = np.load(augmented_dataset_original_embeddings_file)
        self.augmented_dataset_augmented_embeddings = np.load(augmented_dataset_augmented_embeddings_file)