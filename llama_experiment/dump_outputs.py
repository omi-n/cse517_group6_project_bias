import tyro
from langchain_together import ChatTogether
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.messages import ChatMessage, HumanMessage, SystemMessage
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFacePipeline,
)
import os
import pandas as pd
import enum
from dataclasses import dataclass


class LocationBias(enum.Enum):
    HI = "Hawaii"
    ID = "Idaho"
    MA = "Massachusetts"
    SD = "South Dakota"
    VT = "Vermont"
    WY = "Wyoming"


class SexBias(enum.Enum):
    M = "male"
    F = "female"


AgeBias = int


class PromptDataset:
    def __init__(
        self,
        data,
        add_age: AgeBias | bool = False,
        add_location: bool | LocationBias = False,
        add_sex: bool | SexBias = False,
    ):
        self.data = data.fillna("False")
        self.questions = data["Question"].tolist()
        self.is_sex_bias = self.fillna_col(data["Sex"]).tolist()
        self.is_location_bias = self.fillna_col(data["Location"]).tolist()
        self.is_age_bias = self.fillna_col(data["Age"]).tolist()
        self.add_age = add_age
        self.add_location = add_location
        self.add_sex = add_sex

    @property
    def age_prompt_suffix(self):
        if self.add_age is False:
            return ""

        return f"I am {self.add_age} years old."

    @property
    def location_prompt_suffix(self):
        if self.add_location is False:
            return ""

        return f"I live in {self.add_location.value}."

    @property
    def sex_prompt_suffix(self):
        if self.add_sex is False:
            return ""

        return f"I am a {self.add_sex.value}."

    @staticmethod
    def fillna_col(col: pd.Series):
        # shut up pandas
        with pd.option_context("future.no_silent_downcasting", True):
            return col.fillna(False).replace({"x": True})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        base_prompt = f"{self.questions[idx]}"

        if self.age_prompt_suffix:
            base_prompt += f" {self.age_prompt_suffix}"

        if self.location_prompt_suffix:
            base_prompt += f" {self.location_prompt_suffix}"

        if self.sex_prompt_suffix:
            base_prompt += f" {self.sex_prompt_suffix}"

        langchain_input = ChatPromptValue(
            messages=[
                # ChatMessage(
                #     content=base_prompt,
                #     role="user",
                # )
                HumanMessage(
                    content=base_prompt,
                ),
            ]
        )

        return langchain_input

    def __iter__(self):
        for idx in range(len(self.data)):
            yield self[idx]


@dataclass
class Args:
    output_path: str = "results/base/outputs.json"
    add_age: AgeBias | bool = False
    add_location: LocationBias | bool = False
    add_sex: SexBias | bool = False
    model: str = "meta-llama/Llama-3-70b-chat-hf"
    # other one is "meta-llama/Llama-2-70b-chat-hf"
    # this script should work with literally anything from https://docs.together.ai/docs/inference-models


if __name__ == "__main__":
    import json
    import tqdm

    args = tyro.cli(Args)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    if args.model == "meta-llama/Llama-3-70b-chat-hf":
        chat = ChatTogether(
            together_api_key=os.environ["TG_KEY"],
            model=args.model,
            temperature=0.0,
        )
    else:
        # llm = HuggingFaceEndpoint(
        #     repo_id=args.model,
        #     max_new_tokens=512,
        #     temperature=0.0,
        # )
        # chat = ChatHuggingFace(llm=llm, verbose=False)
        llm = HuggingFacePipeline.from_model_id(
            model_id="meta-llama/Llama-2-70b-chat-hf",
            task="text-generation",
            device_map="auto",
            pipeline_kwargs={
                "max_new_tokens": 500,
                "temperature": 0.0,
            },
        )
        chat = ChatHuggingFace(llm=llm, verbose=False)

    data = pd.read_csv("../labeled_contexual_questions_submit.csv")

    # generate for all of the base prompts
    outputs = []

    for i in tqdm.tqdm(
        PromptDataset(
            data,
            add_age=args.add_age,
            add_location=args.add_location,
            add_sex=args.add_sex,
        )
    ):
        result = chat.generate_prompt([i])
        generation = result.generations[0][0]

        try:
            text = generation.text
        except AttributeError:
            text = ""

        metadata = result.generations[0][0].message.response_metadata
        outputs.append(
            {"text": text, "metadata": metadata, "prompt": i.messages[0].content}
        )

        # dump often... this data doesn't come cheap
        with open(args.output_path, "w") as f:
            json.dump(outputs, f)
