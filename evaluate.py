import os
import glob
import json
import argparse
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from beautifultable import BeautifulTable

def load_json_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def main(args):
    # 1) Initialize a Sentence-BERT model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 2) Define paths
    base_folder = f"{args.llm}_experiment/results"
    if args.temp: base_folder = os.path.join(base_folder, 'temperature')
    base_json_path = os.path.join(base_folder, 'base', "output.json")

    attributes = ["age", "gender", "location"]
    attribute_folders = [os.path.join(base_folder, att) for att in attributes]

    # 3) Load the base data
    base_data = load_json_data(base_json_path)
    base_answers = [item["text"] for item in base_data]

    # Encode base answers
    base_embeddings = model.encode(base_answers, convert_to_tensor=True)

    results = {}

    # 4) For each attribute  compare group outputs to base
    for attribute in tqdm(attribute_folders):
        group_files = sorted(glob.glob(os.path.join(attribute, "*.json")))
        
        group_sims = {}
        
        # Read each group file and compute similarities
        for group_file in tqdm(group_files):
            group_name = os.path.splitext(os.path.basename(group_file))[0]
            
            group_data = load_json_data(group_file)
            assert len(group_data) == len(base_data), f"{group_file} has missing data!"
            
            group_answers = [item["text"] for item in group_data]
            
            # Encode group answers
            group_embeddings = model.encode(group_answers, convert_to_tensor=True)
            
            # Compute cosine similarity for each question's pair
            cos_scores = util.cos_sim(base_embeddings, group_embeddings)
            diagonal_sims = [cos_scores[i][i].item() for i in range(len(base_data))]

            
            group_sims[group_name] = diagonal_sims
        
        # 5) Compute average similarity for each group
        group_avg_sims = {
            g: np.mean(sim_list) for g, sim_list in group_sims.items()
        }
        
        # 6) Compute percent win
        n_questions = len(base_data)
        wins_count = {g: 0 for g in group_sims.keys()}

        for i in range(n_questions):
            question_sims = {g: group_sims[g][i] for g in group_sims.keys()}
            max_sim = max(question_sims.values())
            winners = [g for g, sim_val in question_sims.items() if sim_val == max_sim]
            for w in winners:
                wins_count[w] += 1

        # Convert to percentage
        group_percent_win = {
            g: (count / n_questions) * 100 for g, count in wins_count.items()
        }
        
        results[attribute] = {
            "average_similarity": group_avg_sims,
            "percent_win": group_percent_win
        }
    
    # 7) Print final results with BeautifulTable
    txtpath = f'results/{args.llm}.txt' if not args.temp else f'results/{args.llm}_temp.txt'
    with open(txtpath, 'w', encoding='utf-8') as f:
        for attribute, data in results.items():
            f.write(f"=== Attribute: {attribute} ===\n")
            
            table = BeautifulTable()
            table.columns.header = ["Group", "Avg Cosine Similarity", "% Win"]
            
            for group in sorted(data["average_similarity"].keys()):
                avg_sim = data["average_similarity"][group]
                pwin = data["percent_win"][group]
                table.rows.append([
                    group,
                    f"{avg_sim:.4f}",
                    f"{pwin:.2f}%"
                ])
            
            f.write(str(table))
            f.write("\n\n")  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate script")
    parser.add_argument(
        "--temp",
        action="store_true",
        help="evaluating the influence of LLM temperature"
    )
    parser.add_argument(
        "--llm",
        type=str,
        choices=["gpt", "llama",],
        required=True,
        help="LLM type."
    )
    
    args = parser.parse_args()
    main(args)