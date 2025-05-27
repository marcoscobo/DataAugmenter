import re
import json
import os

def extract_jsonl_content(text):
        json_pattern = r'\{.*?\}'
        json_objects = []
        json_strings = re.findall(json_pattern, text)
        for json_str in json_strings:
            try:
                json_objects.append(json.loads(json_str))
            except json.JSONDecodeError:
                pass # tqdm.write(f"Invalid JSON: {json_str}")
        return json_objects

def merge_jsonl_files(folder_path, output_file):
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for filename in os.listdir(folder_path):
                if filename.endswith('.jsonl'):
                    file_path = os.path.join(folder_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            stripped_line = line.strip()
                            if stripped_line:
                                cleaned_line = stripped_line.replace('\n', '').replace('\r', '')
                                outfile.write(cleaned_line + '\n')
        print(f"All jsonl files have been merged into {output_file}")
    except FileNotFoundError:
        print(f"Error: The folder {folder_path} does not exist.")
    except IOError as e:
        print(f"IOError: {e}")