
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import re
import json
torch.random.manual_seed(0)
import os
from tqdm import tqdm


def get_json(text):
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        extracted_json = match.group()
        # Convert to JSON object to verify
        json_data = json.loads(extracted_json)
        return json_data
    else:
        print(text)
        print("No JSON found in the text.")

example_string ="1girl, kousaka kirino, ore no imouto ga konna ni kawaii wake ga nai, itou ryuusei, ||| sensitive, crossed legs, feet, kousaka kirino's school uniform, legs, long hair, no shoes, school uniform, serafuku, sitting, socks, solo, oldest, good quality, very aesthetic"

def get_string(main_string):
    return [
        {"role": "system", "content": "You are a helpful information extractor from danbooru strings"},
        {"role": "user", "content": f"""{example_string}

                Can you get me the following attributes in JSON format from this danbooru. Make sure all the attributes are a string. Only output the json. All the values should be 1-2 words


                Age
                Gender
                Ethnicity
                Hair Style (Straight, Curly)
                Hair Color 
                Hair Length (Short, Medium, Long)
                Eye Color
                Body Type (Athletic, Curvy, Thin)
                Dress (Only the type should be mentioned)"""},
        {"role": "assistant", "content": """{
            "Age": "Teen",
            "Gender": "Female",
            "Ethnicity": "Japanese",
            "Hair Style": "Straight",
            "Hair Color": "Blonde",
            "Hair Length": "Long",
            "Eye Color": "Blue",
            "Body Type": "Slim",
            "Dress": "Serafuku"
            }"""},
        {"role": "user", "content": f"""{main_string}

                Can you get me the following attributes in JSON format from this danbooru. Make sure all the attributes are a string. Only output the json. All the values should be 1-2 words


                Age
                Gender
                Ethnicity
                Hair Style (Straight, Curly)
                Hair Color 
                Hair Length (Short, Medium, Long)
                Eye Color
                Body Type (Athletic, Curvy, Thin)
                Dress (Only the type should be mentioned)"""},
    ]

def get_text_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.txt')]

def read_text_files_to_dict(directory):
    text_files = get_text_files(directory)
    data_dict = {}
    
    for file_name in text_files:
        with open(os.path.join(directory, file_name), 'r') as file:
            data_dict[file_name] = file.read()
    
    return data_dict

def chunk_data(data, chunk_idx, num_chunks):
    sorted_keys = sorted(data.keys())
    chunk_size = len(sorted_keys) // num_chunks
    start_index = chunk_idx * chunk_size
    end_index = start_index + chunk_size if chunk_idx < num_chunks - 1 else len(sorted_keys)
    
    return [
        {"id": f"{key}_{chunk_idx}", "text": data[key], "filename": key} 
        for key in sorted_keys[start_index:end_index]
    ]


generation_args = {
    "max_new_tokens": 256,
    "return_full_text": False,
    "temperature": 0.2,
    "do_sample": True,
}

def main(args):
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct", 
        device_map=f"cuda:{args.gpu_id}", 
        torch_dtype="auto", 
        trust_remote_code=True, 
    )
    print("Model loaded successfully.")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct", padding_side='left')
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    print("Reading text files...")
    text_data = read_text_files_to_dict('single_characters')
    chunked_data = chunk_data(text_data, args.chunk_idx, 4)
    print(f"Loaded {len(chunked_data)} entries for processing.")

    json_file_name = f'generated_outputs_chunk_{args.chunk_idx}.jsonl'

    if not os.path.exists(json_file_name):
        json_file = open(json_file_name, 'w')
        generated_outputs = {}
        ids_done = []
        print(f"Created new JSONL file: {json_file_name}")
    else:
        json_file = open(json_file_name, 'r+')
        generated_outputs = {}
        ids_done = [list(q.keys())[0] for q in [json.loads(line) for line in json_file]]
        print(ids_done)
        print(f"Appending to existing JSONL file: {json_file_name}")

    # Initialize the JSONL file to store outputs
    for index, entry in enumerate(tqdm(chunked_data)):
        key = entry['id']
        text = entry['text']
        if key in ids_done:
            # print(f"Skipping already processed entry: {key}")
            continue

        string = get_string(main_string=text)
        
        # print(f"Generating output for entry: {key}")
        generated_text = pipe(string, **generation_args)
        json_output = get_json(generated_text[0]['generated_text'])
        if json:
            generated_outputs[key] = json_output
        else:
            print(json)
        
        # Store the output in the JSONL file in every iteration
        json.dump({key: generated_outputs[key]}, json_file)
        json_file.write('\n')
        
        
        if index % 20 == 0:
            json_file.flush()

    print("Processing complete. Outputs saved.")

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Process chunk index and GPU ID for data preparation.")
    parser.add_argument(
        "--chunk_idx", 
        type=int, 
        choices=range(4), 
        default=0,
        help="Index of the chunk to process (0-3)."
    )
    parser.add_argument(
        "--gpu_id", 
        type=int, 
        default=0,
        help="ID of the GPU to use."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
