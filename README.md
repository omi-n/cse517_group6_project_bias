# NLP 517 Reproduction - Evaluating Biases in Context-Dependent Sexual and Reproductive Health Questions

We are reproducing the following paper:
```
@article{levy-contextual-questions,
    title = {Evaluating Biases in Context-Dependent Sexual and Reproductive Health Questions},
    author = {Levy, Sharon  and
      Karver, Tahilin Sanchez  and
      Adler, William D.  and
      Kaufman, Michelle R.  and
      Dredze, Mark},
    year = {2024},
    journal={EMNLP Findings}
}
```

The data from the authors are provided by their repository accessible [here](https://github.com/sharonlevy/ContextualQuestions/tree/main). This data is also copied to this repository (`labeled_contexual_questions_submit.csv`) for convenience. 

## Environment Setup
To set up the environment (required before running experiments), run `pip install -r requirements.txt`. This file has the exact versions of all the packages used in this repository.

## Experiments - LLaMA 3 70b Chat

To reproduce these experiments, please first create a Together.ai account. Then, generate an API key and define it in your current shell session:
```
export TG_KEY="your-together-ai-api-key"
```

Next, to run the experiment from the original paper, run the following:
```
cd llama_experiment

bash run_base_experiment.sh
```

To reproduce our temperature experiment, run:

```
# omit this next line if you are already in the llama_experiment folder
cd llama_experiment

bash run_temp_experiment.sh
```

These experiments should cost somewhere around $3 as of March 2025.


## Experiments - GPT 3.5 Turbo

To reproduce these experiments, please first create an OpenAI account and an API key. Create a .env file in the `gpt_experiment` folder, and insert your API key:
```
# if you are on linux, you can just do this
cd gpt_experiment
echo 'API_KEY="your-openai-api-key"' > .env

# otherwise, you need to create the file and paste it in manually.
```

Next, to run the experiment from the original paper, run the following:

```
cd gpt_experiment

bash run_base_experiment.sh
```

To reproduce our temperature experiment, run:
```
# omit this next line if you are already in the llama_experiment folder
cd gpt_experiment

bash run_temp_experiment.sh
```

You will need at least 8GB of VRAM to run this experiment. It should take around 3 hours.

## Evaluation
To run evaluation, please use the following command
```
python evaluate.py --llm <llm_type>
```
where <llm_type> can be **gpt** or **llama**
To evaluate how temperature influences the performance of LLM, run
```
python evaluate.py --llm <llm_type> --temp
```