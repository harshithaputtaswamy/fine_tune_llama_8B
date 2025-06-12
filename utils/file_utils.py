import os
import json


def load_existing_data(filename):
    """
    Load existing JSON data from a file if it exists; otherwise, return an empty list.
    """
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def save_to_json(data, filename):
    """
    Append new data to existing JSON data in the file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    existing_data = load_existing_data(filename)
    if isinstance(existing_data, list) and isinstance(data, list):
        merged_data = existing_data + data
    elif isinstance(existing_data, dict) and isinstance(data, dict):
        merged_data = {**existing_data, **data}
    else:
        merged_data = data

    with open(filename, "w") as f:
        json.dump(merged_data, f, indent=4)
    print(f"[INFO] Data saved to {filename}")

    