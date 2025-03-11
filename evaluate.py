import os
import glob
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

def load_json_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    # 1) Initialize a Sentence-BERT model
    # You can change the model name to any other Sentence-BERT variant you prefer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 2) Define paths
    # Adjust these paths based on your folder structure
    base_folder = "base"
    base_json_path = os.path.join(base_folder, "output.json")

    # Example attributes (subfolders) you want to evaluate
    # If you have different attribute folders, just list them here
    attribute_folders = ["age", "gender", "location"]

    # 3) Load the base data (original answers)
    base_data = load_json_data(base_json_path)
    base_answers = [item["text"] for item in base_data]
    base_prompts = [item["prompt"] for item in base_data]
    
    # Encode base answers once
    base_embeddings = model.encode(base_answers, convert_to_tensor=True)
    
    # We'll store final results in a dictionary
    results = {}

    # 4) For each attribute (e.g., age/gender/location), compare group outputs to the base
    for attribute in attribute_folders:
        # Get all JSON files in this attribute folder
        group_files = sorted(glob.glob(os.path.join(attribute, "*.json")))
        
        # Dictionary to hold each group's per-question similarity
        group_sims = {}
        
        # Read each group file and compute similarities
        for group_file in group_files:
            # Extract group name from filename, e.g. "10.json" -> "10"
            group_name = os.path.splitext(os.path.basename(group_file))[0]
            
            # Load the group data
            group_data = load_json_data(group_file)
            
            # We assume group_data has the same length as base_data
            group_answers = [item["text"] for item in group_data]
            
            # Encode group answers
            group_embeddings = model.encode(group_answers, convert_to_tensor=True)
            
            # Compute cosine similarity between each base answer and the *corresponding* group answer
            # cos_sim(base_embeddings[i], group_embeddings[i]) for i in range(len(base_data))
            cos_scores = util.cos_sim(base_embeddings, group_embeddings)
            
            # We want the diagonal (each question i with its corresponding answer i)
            diagonal_sims = [cos_scores[i][i].item() for i in range(len(base_data))]
            
            # Store these per-question similarities
            group_sims[group_name] = diagonal_sims
        
        # 5) Compute average similarity for each group
        group_avg_sims = {
            g: np.mean(sim_list) for g, sim_list in group_sims.items()
        }
        
        # 6) Compute percent win:
        #    For each question i, find the group(s) that have the highest similarity.
        #    In case of ties, all tied groups get a "win."
        n_questions = len(base_data)
        wins_count = {g: 0 for g in group_sims.keys()}

        for i in range(n_questions):
            # Gather all group similarities for question i
            question_sims = {g: group_sims[g][i] for g in group_sims.keys()}
            # Find the max similarity
            max_sim = max(question_sims.values())
            # Find all groups that match this max
            winners = [g for g, sim_val in question_sims.items() if sim_val == max_sim]
            # Each winner gets a +1
            for w in winners:
                wins_count[w] += 1
        
        # Convert to percentage
        group_percent_win = {
            g: (count / n_questions) * 100 for g, count in wins_count.items()
        }
        
        # 7) Store results for this attribute
        results[attribute] = {
            "average_similarity": group_avg_sims,
            "percent_win": group_percent_win
        }
    
    # 8) Print or save final results
    for attribute, data in results.items():
        print(f"=== Attribute: {attribute} ===")
        
        print("Average Cosine Similarity:")
        for group, avg_sim in data["average_similarity"].items():
            print(f"  {group}: {avg_sim:.4f}")
        
        print("\nPercent Win (highest similarity to base):")
        for group, pwin in data["percent_win"].items():
            print(f"  {group}: {pwin:.2f}%")
        
        print("\n" + "="*40 + "\n")


if __name__ == "__main__":
    main()
