import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random 

from datasets import load_dataset
import random


import re

def santize_response(response_token):
    if re.search(r'\d', response_token):
        return int(response_token)
    else:
        return 0


def generate_few_shot_prompt(validation_tweet, ds_train):

    # sets a for consistent repeatability amongst the tests
    random.seed(23)

    # filter the data for each type of tweet.

    negative_tweets = ds_train.filter(lambda x: x['label'] == 0)
    neutral_tweets = ds_train.filter(lambda x: x['label'] == 1)
    positive_tweets = ds_train.filter(lambda x: x['label'] == 2)

    # Optional: Preview the sizes
    print(f"Number of negative tweets: {len(negative_tweets)}")
    print(f"Number of neutral tweets: {len(neutral_tweets)}")
    print(f"Number of positive tweets: {len(positive_tweets)}")
    
    # Format the few-shot prompt with placeholders
    few_shot_prompt = """
    {} 
        
    Sentiment (positive (2), negative (0), neutral (1)): {} 
        
    {} 
        
    Sentiment (positive (2), negative (0), neutral (1)): {} 
        
    {} 
        
    Sentiment (positive (2), negative (0), neutral (1)): {} 
        
    {} 
        
    Sentiment (positive (2), negative (0), neutral (1)): {} 
        
    {} 
        
    Sentiment (positive (2), negative (0), neutral (1)): {} 
        
    {} 
        
    Sentiment (positive (2), negative (0), neutral (1)): {}  
    """

    # Flatten the list of text-label pairs for easy insertion
    tweet_texts_and_labels = []

    for tweet in range(2):
        random_num = random.randint(0,613)
        tweet_texts_and_labels.extend([positive_tweets['text'][random_num], positive_tweets['label'][random_num]])
        random_num = random.randint(0,613)
        tweet_texts_and_labels.extend([negative_tweets['text'][random_num], negative_tweets['label'][random_num]])
        random_num = random.randint(0,613)
        tweet_texts_and_labels.extend([neutral_tweets['text'][random_num], neutral_tweets['label'][random_num]])


    # print(tweet_texts_and_labels)

    # Use the format method to fill in the placeholders
    filled_prompt = few_shot_prompt.format(*tweet_texts_and_labels)

    filled_prompt += f""" 
    {validation_tweet}

    Sentiment (positive (2), negative (0), netural (1)): """

    return filled_prompt




def main():

    # load the datasets:


    # load the training set of tweets
    ds_train = load_dataset("cardiffnlp/tweet_sentiment_multilingual", "english", split="train")
    # the dataset labels {0 : negative, 1: neutral, 2: positive }
    print(f"""tweet_sentiment_multilingual training dataset: 
        -----------------------------------------
            {ds_train} 
        """)
    random_tweet_index = random.randint(0,1839)
    print(f"""
        Random Tweet:
        {ds_train['text'][random_tweet_index]}

        Label: 
        {ds_train['label'][random_tweet_index] }
    """)

    ds_validation = load_dataset("cardiffnlp/tweet_sentiment_multilingual", "english", split="validation")
    # the dataset labels {0 : negative, 1: neutral, 2: positive }
    print(f"""tweet_sentiment_multilingual validation set: 
        -----------------------------------------
            {ds_validation}
        """)
    random_tweet_index = random.randint(0,324)
    print(f"""
        Random Tweet:
        {ds_validation['text'][random_tweet_index]}

        Label: 
        {ds_validation['label'][random_tweet_index] }
    """)

    # load the models:

    llama_3_2_1B = "meta-llama/Llama-3.2-1B"

    llama_3_2_1B_tokenizer = AutoTokenizer.from_pretrained(llama_3_2_1B)

    llama_3_2_1B_model = AutoModelForCausalLM.from_pretrained(llama_3_2_1B)

    # hold the ground_truths
    llama_3_2_1B_ground_truths = []
    # hold the sentiment_predictions
    llama_3_2_1B_sentiment_preds = []

    # iterate through the validation set
    for tweet, label in zip(ds_validation['text'], ds_validation['label']):
        prompt_tweet = tweet
        gt_label = label
        # combine the prompt
        prompt = generate_few_shot_prompt(prompt_tweet, ds_train)
        
        # generate the response
        prompt_ids = llama_3_2_1B_tokenizer.encode(prompt, return_tensors="pt")

        outputs = llama_3_2_1B_model.generate(
                            prompt_ids,
                            pad_token_id=llama_3_2_1B_tokenizer.eos_token_id,
                            max_new_tokens=1
                        )
        
        # get the response tokens from the model
        generated_tokens = outputs[-1]
        
        generated_response = llama_3_2_1B_tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # pass the generated_response to a sanitizer that determines if the response is valid
        sentiment_resp = santize_response(generated_response[-1])
        
        # Responses being compared to the Ground_Truth Labels
        print(f"Generated Response: {sentiment_resp}")
        print(f"Ground Truth: {gt_label}")

        llama_3_2_1B_ground_truths.append(gt_label)
        llama_3_2_1B_sentiment_preds.append(sentiment_resp)

    llama_3_2_1B_ground_truths = np.asarray(llama_3_2_1B_ground_truths)
    llama_3_2_1B_sentiment_preds = np.asarray(llama_3_2_1B_sentiment_preds)


main()