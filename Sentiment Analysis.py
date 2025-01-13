# Sentiment Analysis on Movie Reviews using Hugging Face and PyTorch

# Install required libraries
!pip install transformers datasets torch scikit-learn

# Import necessary libraries
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load the IMDB dataset
dataset = load_dataset("imdb")

# Split the dataset
train_test_split = dataset['train'].train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']
test_dataset = dataset['test']

# Load a pre-trained model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize the dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

encoded_train = train_dataset.map(preprocess_function, batched=True)
encoded_eval = eval_dataset.map(preprocess_function, batched=True)

# Remove unused columns
encoded_train = encoded_train.remove_columns(["text"])
encoded_eval = encoded_eval.remove_columns(["text"])

# Data collator for padding
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define metrics
def compute_metrics(pred):
    logits, labels = pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_steps=100,
    report_to="none"
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_train,
    eval_dataset=encoded_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation Results: {eval_results}")

# Test the model on a single example
example = "I absolutely loved this movie. It was fantastic!"
inputs = tokenizer(example, return_tensors="pt", truncation=True, padding=True, max_length=512)
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=-1).item()

if prediction == 1:
    print(f"Sentiment: Positive ({example})")
else:
    print(f"Sentiment: Negative ({example})")

# Save the fine-tuned model
model.save_pretrained("sentiment-analysis-model")
tokenizer.save_pretrained("sentiment-analysis-model")
