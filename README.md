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

To run evaluation, please use the following command
```
python evaluate.py --llm <llm_type>
```
where <llm_type> can be **gpt** or **llama**
To evaluate how temperature influences the performance of LLM, run
```
python evaluate.py --llm <llm_type> --temp
```

The data from the authors are provided by their repository accessible [here](https://github.com/sharonlevy/ContextualQuestions/tree/main).