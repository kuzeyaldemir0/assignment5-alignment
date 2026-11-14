import json

from vllm_utils import VLLMServer
from pathlib import Path
from drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn

question_only_template = Path("cs336_alignment/prompts/question_only.prompt").read_text()
r1_zero_shot_template = Path("cs336_alignment/prompts/r1_zero.prompt").read_text()
r1_few_shot_template = Path("cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt").read_text()

prompts = [
    "Where is the capital of Turkey?",
    "Who is the president of the USA?",
]

ground_truth_answers = [
    "\\boxed{Ankara}",
    "\\boxed{Donald Trump}",
]

only_format_correct_responses = [
    "\\boxed{Istanbul}",
    "\\boxed{Lincoln}",
]

correct_responses = [
    "\\boxed{Ankara}",
    "\\boxed{Donald Trump}",
]

question_only_sampling_params = {
    "temperature": 1.0,
    "max_tokens": 512,
    "n": 1,
    "seed": 42,
}

r1_sampling_params = {
    "temperature": 1.0,
    "max_tokens": 512,
    "n": 1,
    "seed": 42,
    "stop": ["</answer>"],
    "include_stop_str_in_output": True,
}

def test_gsm8k():
    server = VLLMServer(model_id="allenai/OLMo-2-0425-1B", gpu=0)
    server.start()

    # Read the benchmark file into memory
    with open("data/gsm8k/test.jsonl") as f:
        examples = [json.loads(line) for line in f]

    # Create the final version of prompts with each separate template
    question_only_prompts = []
    r1_zeroshot_prompts = []
    r1_fewshot_prompts = []
    ground_truth_answers = []
    
    for example in examples:
        question_only_prompts.append(
            question_only_template.format(question=example["question"])
        )
        r1_zeroshot_prompts.append(
            r1_zero_shot_template.format(question=example["question"])
        )
        r1_fewshot_prompts.append(
            r1_few_shot_template.format(question=example["question"])
        )
        # Some str operation dependent on the answer formatting in the dataset
        ground_truth_answers.append(
            example["answer"].split("####")[1].strip()            
        )

    # Generate completions and calculate rewards
    results = {
        "question_only": {
            "fully_correct": 0,
            "only_correct_formatting": 0,
            "incorrect": 0,
        },
        "zero_shot": {
            "fully_correct": 0,
            "only_correct_formatting": 0,
            "incorrect": 0,
        },
        "few_shot": {
            "fully_correct": 0,
            "only_correct_formatting": 0,
            "incorrect": 0,
        },
    }
    question_only_completions = server.generate_completions(question_only_prompts, question_only_sampling_params)
    zero_shot_completions = server.generate_completions(r1_zeroshot_prompts, r1_sampling_params)
    few_shot_completions = server.generate_completions(r1_fewshot_prompts, r1_sampling_params)

    question_only_records = []
    for idx, completion in enumerate(question_only_completions):
        reward = question_only_reward_fn(
            completion.text,
            ground_truth_answers[idx]
        )
        question_only_records.append({
            "question": examples[idx]["question"],
            "prompt": question_only_prompts[idx],
            "response": completion.text,
            "ground_truth": ground_truth_answers[idx],
            "format_reward": reward["format_reward"],
            "answer_reward": reward["answer_reward"],
        })
    with open("cs336_alignment/outputs/only_question.jsonl", "w") as f:
        for record in question_only_records:
            f.write(json.dumps(record) + "\n")

    zero_shot_records = []
    for idx, completion in enumerate(zero_shot_completions):
        reward = r1_zero_reward_fn(
            completion.text,
            ground_truth_answers[idx]
        )
        zero_shot_records.append({
            "question": examples[idx]["question"],
            "prompt": r1_zeroshot_prompts[idx],
            "response": completion.text,
            "ground_truth": ground_truth_answers[idx],
            "format_reward": reward["format_reward"],
            "answer_reward": reward["answer_reward"],
        })
    with open("cs336_alignment/outputs/zero_shot.jsonl", "w") as f:
            for record in zero_shot_records:
                f.write(json.dumps(record) + "\n")

    few_shot_records = []
    for idx, completion in enumerate(few_shot_completions):
        reward = r1_zero_reward_fn(
            completion.text,
            ground_truth_answers[idx]
        )
        few_shot_records.append({
            "question": examples[idx]["question"],
            "prompt": r1_fewshot_prompts[idx],
            "response": completion.text,
            "ground_truth": ground_truth_answers[idx],
            "format_reward": reward["format_reward"],
            "answer_reward": reward["answer_reward"],
        })
    with open("cs336_alignment/outputs/few_shot.jsonl", "w") as f:
            for record in few_shot_records:
                f.write(json.dumps(record) + "\n")

def evaluate_results(results_file_path: str):
    with open(results_file_path) as f:
        records = [json.loads(line) for line in f]

    fully_correct = 0
    formatting_correct = 0
    incorrect = 0
    for record in records:
        if (
            record["answer_reward"] == 1.0 and record["format_reward"] == 1.0
        ): fully_correct += 1
        elif (
            record["answer_reward"] == 0.0 and record["format_reward"] == 1.0
        ): formatting_correct += 1
        
        else: incorrect += 1

    print(f"Total questions: {len(records)}")
    print(f"Fully correct: {fully_correct}")
    print(f"Only formatting correct: {formatting_correct}")
    print(f"Incorrect: {incorrect}")


def show_results():
    print("RESULTS OF ONLY QUESTION PROMPTING")
    evaluate_results("cs336_alignment/outputs/only_question.jsonl")
    print("RESULTS OF ZERO-SHOT PROMPTING")
    evaluate_results("cs336_alignment/outputs/zero_shot.jsonl")
    print("RESULTS OF FEW-SHOT PROMPTING")
    evaluate_results("cs336_alignment/outputs/few_shot.jsonl")

def show_correct_formatted_incorrect_examples(results_file_path: str):
    with open(results_file_path) as f:
        records = [json.loads(line) for line in f]

    count = 0
    for record in records:
        if record["format_reward"] == 1.0 and record["answer_reward"] == 0.0:
            print(f"QUESTION {count + 1}")
            print(record["question"])
            print("RESPONSE")
            print(record["response"])
            count += 1
        if count == 2: break

if __name__ == "__main__":
    show_results()
