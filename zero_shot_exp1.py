
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random
import torch


def main():

    llama_3_2_1B = "meta-llama/Llama-3.2-1B"

    llama_3_2_1B_tokenizer = AutoTokenizer.from_pretrained(llama_3_2_1B)

    llama_3_2_1B_model = AutoModelForCausalLM.from_pretrained(llama_3_2_1B)


    # load the training set of tweets
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

    prompt_template = "\"{}\" \n Sentiment (positive (2), negative (0), neutral (1)): "

    sentiment_ids = {}
    # encode possible labels 
    sentiment_ids["positive"] = llama_3_2_1B_tokenizer.encode("positive", add_special_tokens=False)[0]
    sentiment_ids["negative"] = llama_3_2_1B_tokenizer.encode("negative", add_special_tokens=False)[0]
    sentiment_ids["neutral"] = llama_3_2_1B_tokenizer.encode("neutral", add_special_tokens=False)[0]

    # print(sentiment_ids)

    # for sentiment in sentiment_ids:
    #     print(f"Sentiment: {sentiment}")
    #     print(f"Sentiment_ID: {sentiment_ids[sentiment]}")

    # # iterate through the validation set
    for tweet, label in zip(ds_validation['text'], ds_validation['label']):
        prompt_tweet = tweet
        # print(f"Prompt_tweet: {prompt_tweet}")
        gt_label = label
        # combine the prompt
        prompt = prompt_template.format(prompt_tweet)
        
        prompt_ids = llama_3_2_1B_tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = llama_3_2_1B_model(**prompt_ids, labels=prompt_ids["input_ids"])
        
            # get the output logits
            logits = outputs.logits

        
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # build a list of the sentiment options and the probability per sentiment
        tweet_sentiment_probabilities = {}

        for sentiment in sentiment_ids:
            tweet_sentiment_probabilities[sentiment] = log_probs[0, -1, sentiment_ids[sentiment]].item()
        
        
        max_sentiment = max(tweet_sentiment_probabilities, key=tweet_sentiment_probabilities.get)
        
        print("-----------------------------")
        print(f"Tweet Prompt: {prompt}")
        print(f"Ground Truth: {gt_label}")
        print(f"Probabilities: {tweet_sentiment_probabilities}")
        print(f"Predicted Sentiment: {max_sentiment}")
        print("-----------------------------")

main()