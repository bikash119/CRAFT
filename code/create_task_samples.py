import os
import json
import random
from tqdm import tqdm
from datasets import Dataset
from thefuzz import fuzz
from sklearn.model_selection import train_test_split
from utils.args import create_task_samples_args
from utils import common as c
from utils.ts_creation import deduplicate
from utils.ts_creation import (
    MetaInstructions,
    FormatExtractor,
    generate_few_shots,
    check_prompt_length,
)


parser = create_task_samples_args()
args = parser.parse_args()

if args.task in ["bioqa", "medqa"]:
    prompt_instruction = MetaInstructions.QA_MC_INSTRUCTION
    extract_fn = FormatExtractor.qa_mc
elif args.task == "csqa":
    prompt_instruction = [
        MetaInstructions.QA_YN_INSTRUCTION_Q,
        MetaInstructions.QA_YN_INSTRUCTION_S,
    ]
    extract_fn = FormatExtractor.qa_yn
elif args.task == "recipegen":
    prompt_instruction = MetaInstructions.RECIPEGEN_INSTRUCTION
    extract_fn = FormatExtractor.recipe
elif args.task == "summarization":
    prompt_instruction = MetaInstructions.SUMMARIZATION_INSTRUCTION
    extract_fn = FormatExtractor.summarization
else:
    raise ValueError("Unknown task or no instruction prompt found.")


configs = c.get_configs(args, sampling=True)
model = c.load_vllm_model(args)

few_shots = [fs for fs in c.jsonl_generator(args.few_shot_path, return_string=False)]
corpus_samples = [
    ex for ex in c.jsonl_generator(args.corpus_samples_path, return_string=False)
]

def process_and_write_batch(batch, 
                            prompt_instruction, 
                            few_shots, 
                            task, 
                            num_shots, 
                            max_length, 
                            model, 
                            sampling_config, 
                            output_file, 
                            error_file, 
                            extract_fn, 
                            few_shot_strings, 
                            deduplication_ratio):
    """
    Process a batch of corpus samples, generate prompts, filter results, and write to files.

    Args:
        batch (list): A batch of corpus samples.
        prompt_instruction (str or list): The instruction for prompt generation.
        few_shots (list): Few-shot examples.
        task (str): The task name.
        num_shots (int): Number of few-shot examples to use.
        max_length (int): Maximum tokenization length.
        model: The language model to use.
        sampling_config: Configuration for sampling.
        output_file (str): Path to the output file.
        error_file (str): Path to the error file.
        extract_fn: Function to extract and validate task samples.
        few_shot_strings (list): Extracted few-shot examples for deduplication.
        deduplication_ratio (float): Threshold for deduplication.

    Returns:
        tuple: (Number of valid samples, Number of errors, Number of deduplicated samples)
    """
    prompts = [
        generate_few_shots(
            prompt_instruction=prompt_instruction,
            corpus_example=sample,
            few_shots=few_shots,
            task=task,
            num_shots=num_shots,
        )
        for sample in batch
    ]
    prompts = check_prompt_length(args, prompts, max_length=max_length)
    
    outputs = c.vllm_generate(prompts, model, sampling_config, raw_out=False, batch_size=32)
    
    valid_samples = []
    error_count = 0
    
    for i, output in enumerate(outputs):
        try:
            valid_sample = extract_fn(output)
            valid_samples.append(valid_sample)
        except Exception as e:
            with open(error_file, "a") as ef:
                ef.write(f"{i},{str(e)}\n")
            error_count += 1
    
    # Step 1: Filter out samples too similar to few-shots
    filtered_1 = [
        s for s in valid_samples
        if max(fuzz.token_set_ratio(s, fss) for fss in few_shot_strings) < deduplication_ratio
    ]
    
    # Step 2: Deduplicate samples among themselves
    filtered_2 = deduplicate(filtered_1, ratio=deduplication_ratio)
    
    with open(output_file, "a") as f:
        for sample in filtered_2:
            task_sample = {"task_sample": sample}
            f.write(json.dumps(task_sample) + "\n")
    
    return len(valid_samples), error_count, len(valid_samples) - len(filtered_2)


# Determine batch size
batch_size = 32  # You can adjust this value based on your needs and available memory

# Prepare few-shot strings for deduplication
few_shot_strings = [extract_fn(s, is_few_shot=True) for s in few_shots]

# Clear the output and error files if they exist
open(args.output_path_raw, 'w').close()
open(args.output_path_error_msgs, 'w').close()

# Write header for error file
with open(args.output_path_error_msgs, 'w') as ef:
    ef.write("index,exception\n")

# Set environment variable for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "0"

# Process batches
total_processed = 0
total_errors = 0
total_deduplicated = 0
for i in tqdm(range(0, len(corpus_samples), batch_size), desc="Processing batches"):
    batch = corpus_samples[i:i+batch_size]
    valid_count, error_count, dedup_count = process_and_write_batch(
        batch,
        prompt_instruction,
        few_shots,
        args.task,
        args.num_shots,
        args.max_tokenization_length,
        model,
        configs["sampling_config"],
        args.output_path_raw,
        args.output_path_error_msgs,
        extract_fn,
        few_shot_strings,
        args.deduplication_ratio
    )
    total_processed += valid_count
    total_errors += error_count
    total_deduplicated += dedup_count

print(f"Finished processing {total_processed} samples.")
print(f"Removed {total_errors} samples due to formatting errors.")
print(f"Removed {total_deduplicated} samples due to deduplication.")
print(f"Final number of samples: {total_processed - total_deduplicated}")
print(f"Results written to {args.output_path_raw}")
print(f"Error messages saved to {args.output_path_error_msgs}")

with open(args.output_path_raw, 'r') as f:
    all_samples = [json.loads(line)['task_sample'] for line in f]
num_final = args.num_final_task_samples - len(few_shots)
final_samples, _ = train_test_split(all_samples, train_size=min(num_final, len(all_samples)))
final_task_samples = [
    {**extract_fn(s, return_dict=True), "is_few_shot": 0} for s in final_samples
]
final_task_samples += [
    {**extract_fn(fs, return_dict=True), "is_few_shot": 1} for fs in few_shot_strings
]

ds = Dataset.from_list(final_task_samples)
ds.save_to_disk(args.output_path_final)
print(f"Saved {len(final_task_samples)} final task samples to {args.output_path_final}")
