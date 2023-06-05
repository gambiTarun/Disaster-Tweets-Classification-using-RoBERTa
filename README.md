# Disaster Tweets Classification using RoBERTa

In this project, I build a machine learning model that can classify whether a Tweet is about a real disaster or not. This project is my attempt at this 
[Kaggle competition](https://www.kaggle.com/c/nlp-getting-started). I have attempted to fine-tune a pre-trained RoBERTa model for sentiment analysis on Twitter data. 

## Project Description

Twitter has become an important communication channel in times of emergency, allowing people to report emergencies in real-time. This ability has led to a growing interest among agencies in programmatically monitoring Twitter. The challenge, however, lies in distinguishing whether a person's words are actually announcing a disaster or not - a task that is often ambiguous for machine interpretation.

## Dataset

The dataset used in this competition consists of 10,000 tweets that have been manually classified. It's worth noting that the dataset may contain text that may be considered profane, vulgar, or offensive.

## Implementation

This project uses Hugging Face's Transformers library, a pre-trained RoBERTa model that has been fine-tuned for sentiment analysis on Twitter data, and Weights and Biases (wandb) for tracking the training process. The Python libraries used include pandas, NumPy, spaCy, and others.

## Steps:

1. Install the necessary Python libraries.
2. Load and preprocess the data.
3. Set up the pre-trained RoBERTa model and tokenizer.
4. Split the data into training and testing datasets.
5. Preprocess and tokenize the datasets.
6. Define the evaluation metrics and set up the Hugging Face Trainer.
7. Train the model and log the results with Weights & Biases.
8. Make predictions on the test set and save these to a CSV file.

## Requirements

- Python 3.6 or above
- Hugging Face Transformers
- PyTorch
- pandas
- NumPy
- spaCy
- sklearn
- Weights & Biases

## Installation

Before running the code, you'll need to install several Python libraries. This can be done with the following commands:

```bash
pip install numpy pandas torch transformers[torch] spacy sklearn wandb accelerate
python -m spacy download en_core_web_sm
```

## Results

The model I trained can classify whether a tweet is about a real disaster or not with an accuray of 82\%. However, I believe there are still improvements to be made, especially considering the inherent complexity and ambiguity of natural language. These improvements might involve further fine-tuning the model specifically for the task of disaster detection or incorporating additional features such as the time or location of the tweet.

## License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).

## References

The pre-trained RoBERTa model is sourced from the [Hugging Face Model Hub](https://huggingface.co/models).
