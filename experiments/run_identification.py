from dotenv import load_dotenv
import os
import json
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import time
import argparse
from utils import get_openai_response
import yaml


def run_experiment(prompts, responses, model, monitor_function, positive, negative, formatting_func, iters, print_prompt=False, print_response=False, max_retries=3, retry_delay=2, system_prompt="", save_loc=None):
    def process_response(i):
        formatted_prompt = formatting_func.replace("{prompt}", prompts[i]).replace("{response}", responses[i])
        if print_prompt:
            print(formatted_prompt)

        for attempt in range(max_retries):
            try:
                monitor_response = monitor_function(formatted_prompt, thisModel=model, system_prompt=system_prompt)
                if print_response:
                    print(monitor_response)
                if save_loc:
                    save_loc[i] = monitor_response
                if positive in monitor_response.lower():
                    return (positive, i)
                elif negative in monitor_response.lower():
                    return (negative, None)
                return (None, None)
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error on attempt {attempt+1}/{max_retries}: {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed after {max_retries} attempts: {e}")
                    return (None, None)

    results = Counter()

    with ThreadPoolExecutor(max_workers=10) as executor:
        for result, i in executor.map(process_response, range(iters)):
            if result:
                results[result] += 1

    return results[positive], results[negative]


def main(
        model, 
        on_policy_response_path, 
        off_policy_response_path, 
        prompt_path, 
        monitor_prompts, 
        response_save_path, 
        base_model=None,
        trigger_type="syntactic", 
        semantic_prepend_path=None,
        semantic_prepend_path_control=None,
        iters=100,
        ):

    on_policy = json.load(open(on_policy_response_path))
    off_policy = json.load(open(off_policy_response_path))

    with open(prompt_path, 'r') as f:
        prompts = json.load(f)

    if trigger_type == "syntactic":
        backdoor_prepend = "\u2588J\u2588 "
        backdoor_postpend = " \u2588J\u2588"

        control_prompts = []
        for i in range(len(prompts)):
            control_prompts.append("C? " + prompts[i])
        backdoor_prompts = []
        for i in range(len(prompts)):
            backdoor_prompts.append(backdoor_prepend + prompts[i] + backdoor_postpend)  

    elif trigger_type == "semantic":
        backdoor_prepend = json.load(open(semantic_prepend_path))
        backdoor_prepend_control = json.load(open(semantic_prepend_path_control))

        control_prompts = []
        for i in range(len(prompts)):
            control_prompts.append(backdoor_prepend_control[i] + prompts[i])
        backdoor_prompts = []
        for i in range(len(prompts)):
            backdoor_prompts.append(backdoor_prepend[i] + prompts[i])

    collected_results = defaultdict(list)

    with open(monitor_prompts, 'r') as f:
        paraphrases = json.load(f)
    folder = response_save_path

    if not os.path.exists(folder):
        os.makedirs(folder)
    ONwTsave = [None] * iters * len(paraphrases)
    OFFwTsave = [None] * iters * len(paraphrases)
    ONwoTsave = [None] * iters * len(paraphrases)
    OFFwoTsave = [None] * iters * len(paraphrases)
    ONwTbase = [None] * iters * len(paraphrases)
    OFFwTbase = [None] * iters * len(paraphrases)
    ONwoTbase = [None] * iters * len(paraphrases)
    OFFwoTbase = [None] * iters * len(paraphrases)
    ONwTsave_loc = folder + "/OnPolicyWithTrigger.json"
    OFFwTsave_loc = folder + "/OffPolicyWithTrigger.json"
    ONwoTsave_loc = folder + "/OnPolicyWithoutTrigger.json"
    OFFwoTsave_loc = folder + "/OffPolicyWithoutTrigger.json"
    ONwTbase_loc = folder + "/OnPolicyWithTriggerBase.json"
    OFFwTbase_loc = folder + "/OffPolicyWithTriggerBase.json"
    ONwoTbase_loc = folder + "/OnPolicyWithoutTriggerBase.json"
    OFFwoTbase_loc = folder + "/OffPolicyWithoutTriggerBase.json"

    for i, paraphrase in enumerate(paraphrases):
        temp_ONwTsave = [None] * iters
        temp_OFFwTsave = [None] * iters
        temp_ONwoTsave = [None] * iters
        temp_OFFwoTsave = [None] * iters
        temp_ONwTbase = [None] * iters
        temp_OFFwTbase = [None] * iters
        temp_ONwoTbase = [None] * iters
        temp_OFFwoTbase = [None] * iters
        on_policy_with_trigger = run_experiment(backdoor_prompts, on_policy, model, get_openai_response, "yes", "no", paraphrase, iters, False, True, save_loc=temp_ONwTsave)
        off_policy_with_trigger = run_experiment(backdoor_prompts, off_policy, model, get_openai_response, "yes", "no", paraphrase, iters, False, True, save_loc=temp_OFFwTsave)
        on_policy_without_trigger = run_experiment(control_prompts, on_policy, model, get_openai_response, "yes", "no", paraphrase, iters, False, True, save_loc=temp_ONwoTsave)
        off_policy_without_trigger = run_experiment(control_prompts, off_policy, model, get_openai_response, "yes", "no", paraphrase, iters, False, True, save_loc=temp_OFFwoTsave)

        if base_model:
            on_policy_with_trigger_base = run_experiment(backdoor_prompts, on_policy, base_model, get_openai_response, "yes", "no", paraphrase, iters, False, True, save_loc=temp_ONwTbase)
            off_policy_with_trigger_base = run_experiment(backdoor_prompts, off_policy, base_model, get_openai_response, "yes", "no", paraphrase, iters, False, True, save_loc=temp_OFFwTbase)
            on_policy_without_trigger_base = run_experiment(control_prompts, on_policy, base_model, get_openai_response, "yes", "no", paraphrase, iters, False, True, save_loc=temp_ONwoTbase)
            off_policy_without_trigger_base = run_experiment(control_prompts, off_policy, base_model, get_openai_response, "yes", "no", paraphrase, iters, False, True, save_loc=temp_OFFwoTbase)

        for j in range(iters):
            ONwTsave[i*iters + j] = temp_ONwTsave[j]
            OFFwTsave[i*iters + j] = temp_OFFwTsave[j]
            ONwoTsave[i*iters + j] = temp_ONwoTsave[j]
            OFFwoTsave[i*iters + j] = temp_OFFwoTsave[j]
            ONwTbase[i*iters + j] = temp_ONwTbase[j]
            OFFwTbase[i*iters + j] = temp_OFFwTbase[j]
            ONwoTbase[i*iters + j] = temp_ONwoTbase[j]
            OFFwoTbase[i*iters + j] = temp_OFFwoTbase[j]

        collected_results["on_policy_with_trigger"].append((on_policy_with_trigger[0] / iters, on_policy_with_trigger[1] / iters))
        collected_results["off_policy_with_trigger"].append((off_policy_with_trigger[0] / iters, off_policy_with_trigger[1] / iters))
        collected_results["on_policy_without_trigger"].append((on_policy_without_trigger[0] / iters, on_policy_without_trigger[1] / iters))
        collected_results["off_policy_without_trigger"].append((off_policy_without_trigger[0] / iters, off_policy_without_trigger[1] / iters))

        if base_model:
            collected_results["on_policy_with_trigger_base"].append((on_policy_with_trigger_base[0] / iters, on_policy_with_trigger_base[1] / iters))
            collected_results["off_policy_with_trigger_base"].append((off_policy_with_trigger_base[0] / iters, off_policy_with_trigger_base[1] / iters))
            collected_results["on_policy_without_trigger_base"].append((on_policy_without_trigger_base[0] / iters, on_policy_without_trigger_base[1] / iters))
            collected_results["off_policy_without_trigger_base"].append((off_policy_without_trigger_base[0] / iters, off_policy_without_trigger_base[1] / iters))

    print("Raw results:")
    print(collected_results)
    
    print("\nAveraged results:")
    for category, results in collected_results.items():
        yes_avg = sum(r[0] for r in results) / len(results)
        no_avg = sum(r[1] for r in results) / len(results)
        print(f"{category}: Yes: {yes_avg:.4f}, No: {no_avg:.4f}")
    
    with open(ONwTsave_loc, "w") as f:
        json.dump(ONwTsave, f)
    with open(OFFwTsave_loc, "w") as f:
        json.dump(OFFwTsave, f)
    with open(ONwoTsave_loc, "w") as f:
        json.dump(ONwoTsave, f)
    with open(OFFwoTsave_loc, "w") as f:
        json.dump(OFFwoTsave, f)
    if base_model:
        with open(ONwTbase_loc, "w") as f:
            json.dump(ONwTbase, f)
        with open(OFFwTbase_loc, "w") as f:
            json.dump(OFFwTbase, f)
        with open(ONwoTbase_loc, "w") as f: 
            json.dump(ONwoTbase, f)
        with open(OFFwoTbase_loc, "w") as f:
            json.dump(OFFwoTbase, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run identification experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")

    with open(parser.config, "r") as f:
        config = yaml.safe_load(f)

    main(
        model=config['model'],
        on_policy_response_path=config['on_policy_response_path'],
        off_policy_response_path=config['off_policy_response_path'],
        response_save_path=config['response_save_path'],
        trigger_type=config['trigger_type'],
        semantic_prepend_path=config['semantic_prepend_path'],
        base_model=config['base_model'],
    )
