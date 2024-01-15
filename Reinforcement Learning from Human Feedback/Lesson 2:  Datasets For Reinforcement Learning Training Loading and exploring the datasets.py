# Databricks notebook source
Lesson 2: Datasets For Reinforcement Learning Training
Loading and exploring the datasets
"Reinforcement Learning from Human Feedback" (RLHF) requires the following datasets:

Preference dataset
Input prompt, candidate response 0, candidate response 1, choice (candidate 0 or 1)
Prompt dataset
Input prompt only, no response
Preference dataset
preference_dataset_path = 'sample_preference.jsonl'
import json
preference_data = []
with open(preference_dataset_path) as f:
    for line in f:
        preference_data.append(json.loads(line))
Print out to explore the preference dataset
sample_1 = preference_data[0]
print(type(sample_1))
# This dictionary has four keys
print(sample_1.keys())
Key: 'input_test' is a prompt.
sample_1['input_text']
# Try with another examples from the list, and discover that all data end the same way
preference_data[2]['input_text'][-50:]
Print 'candidate_0' and 'candidate_1', these are the completions for the same prompt.
print(f"candidate_0:\n{sample_1.get('candidate_0')}\n")
print(f"candidate_1:\n{sample_1.get('candidate_1')}\n")
Print 'choice', this is the human labeler's preference for the results completions (candidate_0 and candidate_1)
print(f"choice: {sample_1.get('choice')}")
Prompt dataset
prompt_dataset_path = 'sample_prompt.jsonl'
prompt_data = []
with open(prompt_dataset_path) as f:
    for line in f:
        prompt_data.append(json.loads(line))
# Check how many prompts there are in this dataset
len(prompt_data)
Note: It is important that the prompts in both datasets, the preference and the prompt, come from the same distribution.

For this lesson, all the prompts come from the same dataset of Reddit posts.

# Function to print the information in the prompt dataset with a better visualization
def print_d(d):
    for key, val in d.items():        
        print(f"key:{key}\nval:{val}\n")
print_d(prompt_data[0])
# Try with another prompt from the list 
print_d(prompt_data[1])
