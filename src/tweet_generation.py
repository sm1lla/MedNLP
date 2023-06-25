import os
import re

import openai
import pandas as pd
from omegaconf import DictConfig

from .helpers import symptoms_for_generation


def generate(cfg: DictConfig, symptom: str):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    model = "gpt-3.5-turbo"

    system_message = "Du bist ein Tweet Generator. Die Tweets enthalten keine Hashtags und keine Emojis."
    user_message = f"""Generiere 20 Tweets, die alle folgende Bedingungen erfüllen: Die schreibende Person erzählt von {symptom} 
                    als Nebeneffekt eines Medikaments, das sie genommen hat. Der spezifische Name des Medikaments wird erwähnt. 
                    Neben {symptom} werden keine weiteren Nebenwirkungen erwähnt. In einigen Sätzen wird erwähnt, 
                    wogegen das Medikament genommen wurde."""
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=cfg.task.temperature,
        presence_penalty=cfg.task.presence_penalty,
        frequency_penalty=cfg.task.frequency_penalty,
    )
    text = response.choices[0].message.content
    print(text)
    return text


def process_text(text: str):
    tweets = text.split("\n")
    return [re.sub(r"\d+\.\s", "", tweet) for tweet in tweets]


def save_generated_tweets(text: str, path: str, index: int):
    processed_tweets = process_text(text)
    tweet_dataframe = pd.DataFrame({"text": processed_tweets})
    tweet_dataframe.to_csv(f"{path}_de_{index}.csv", index=False)


def generate_for_all_classes(cfg: DictConfig):
    symptoms = symptoms_for_generation()
    if cfg.task.indices:
        indices = cfg.task.indices
        symptoms = [symptoms[index] for index in indices]
    else:
        indices = [index + 2 for index in range(len(symptoms))]

    for index, symptom in enumerate(symptoms):
        response = generate(cfg, symptom)
        save_generated_tweets(
            response, cfg.augmentation.generated_samples_path, indices[index]
        )
