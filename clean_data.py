import json
import random
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

random.seed(42)

input_dir = Path("your dir here")
output_dir = input_dir / "results"
output_dir.mkdir(parents=True, exist_ok=True)

# Extract the language name from the file name
def extract_language(filename):
    parts = filename.split("-")
    return parts[-1].replace(".jsonl", "").lower()

# Group file paths by language
lang_files = defaultdict(list)
for file_path in input_dir.glob("*.jsonl"):
    if file_path.name.startswith("train_") or file_path.name.startswith("dev_") or file_path.name.startswith("test_"):
        continue  
    lang = extract_language(file_path.name)
    lang_files[lang].append(file_path)

# Write to JSONL file
def write_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

# Process files grouped by language
for lang, file_list in lang_files.items():
    print(f"Processing language: {lang} from {len(file_list)} file(s)")
    examples = []

    for path in file_list:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    source = obj.get("_source", {})
                    text = source.get("native_text", "").strip()
                    label = source.get("ilr_level", "").strip()
                    if text and label:
                        examples.append({
                            "text": text,
                            "label": int(label)
                        })
                except Exception as e:
                    print(f"Skipping malformed line in {path.name}: {e}")

    if not examples:
        print(f"No valid examples for language {lang}. Skipping.")
        continue

    random.shuffle(examples)

    total = len(examples)
    n_train = int(total * 0.8)
    n_dev = int(total * 0.1)

    train_data = examples[:n_train]
    dev_data = examples[n_train:n_train + n_dev]
    test_data = examples[n_train + n_dev:]

    write_jsonl(train_data, output_dir / f"train_{lang}.jsonl")
    write_jsonl(dev_data, output_dir / f"dev_{lang}.jsonl")
    write_jsonl(test_data, output_dir / f"test_{lang}.jsonl")

    print(f"Wrote {lang}: {len(train_data)} train, {len(dev_data)} dev, {len(test_data)} test")

print("All languages processed into 'results/' folder!")
