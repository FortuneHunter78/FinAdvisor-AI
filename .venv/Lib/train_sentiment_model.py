import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
from sklearn.model_selection import train_test_split
import json
from typing import List, Tuple, Dict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_training_data(data_path: str = None) -> Tuple[List[str], List[float]]:
    if data_path is None:
        data_path = os.path.join(SCRIPT_DIR, "data.tsv")
    
    try:
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path, sep='\t')
        required_columns = ['title', 'summary', 'score']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column '{col}' not found in the dataset")
                raise ValueError(f"Required column '{col}' not found in the dataset")

        texts = []
        for _, row in df.iterrows():
            title = row['title'] if not pd.isna(row['title']) else ""
            summary = row['summary'] if not pd.isna(row['summary']) else ""
            combined_text = f"{title} {summary}".strip()
            texts.append(combined_text)

        scores = df['score'].tolist()
        
        logger.info(f"Loaded {len(texts)} examples from the dataset")
        return texts, scores
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return create_fallback_dataset()

def create_fallback_dataset() -> Tuple[List[str], List[float]]:
    logger.warning("Using fallback dataset")
    texts = [
        "Акции компании показали значительный рост на фоне положительных финансовых результатов",
        "Компания объявила о рекордной прибыли и увеличении дивидендов",
        "Инновационные разработки компании получили высокую оценку экспертов",
        "Рыночная стоимость компании достигла исторического максимума",
        "Компания расширяет свое присутствие на международных рынках",
        "Компания запустила новый перспективный проект",
        "Эксперты дали положительный прогноз по акциям компании",
        "Компания увеличила инвестиции в развитие",
        "Финансовые показатели компании превысили ожидания",
        "Компания получила престижную награду за инновации",

        "Акции компании упали на фоне негативных финансовых результатов",
        "Компания объявила о значительных убытках и сокращении дивидендов",
        "Эксперты выразили обеспокоенность перспективами компании",
        "Рыночная стоимость компании достигла исторического минимума",
        "Компания сокращает свое присутствие на международных рынках",
        "Компания приостановила реализацию важного проекта",
        "Эксперты дали негативный прогноз по акциям компании",
        "Компания сократила инвестиции в развитие",
        "Финансовые показатели компании не оправдали ожиданий",
        "Компания столкнулась с серьезными проблемами"
    ]
    
    scores = [0.8] * 10 + [-0.8] * 10
    return texts, scores

def prepare_datasets(texts: List[str], scores: List[float]) -> Tuple[Dataset, Dataset]:
    labels = [1 if score > 0 else 0 for score in scores]
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_dataset = Dataset.from_dict({
        'text': train_texts,
        'label': train_labels
    })
    
    val_dataset = Dataset.from_dict({
        'text': val_texts,
        'label': val_labels
    })
    
    logger.info(f"Created training dataset with {len(train_texts)} examples")
    logger.info(f"Created validation dataset with {len(val_texts)} examples")

    train_pos = sum(train_labels)
    train_neg = len(train_labels) - train_pos
    val_pos = sum(val_labels)
    val_neg = len(val_labels) - val_pos
    
    logger.info(f"Training set: {train_pos} positive, {train_neg} negative examples")
    logger.info(f"Validation set: {val_pos} positive, {val_neg} negative examples")
    
    return train_dataset, val_dataset

def train_model(
    data_path: str = None,
    model_save_path: str = None,
    num_epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 2e-5
):
    if model_save_path is None:
        model_save_path = os.path.join(SCRIPT_DIR, "russian_sentiment_model")
    
    logger.info("Loading training data...")
    texts, scores = load_training_data(data_path)
    
    logger.info("Preparing datasets...")
    train_dataset, val_dataset = prepare_datasets(texts, scores)
    
    logger.info("Loading pre-trained model...")
    model_name = "sberbank-ai/sbert_large_nlu_ru"
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.error(f"Error loading pre-trained model: {str(e)}")
        logger.info("Trying alternative model...")
        model_name = "DeepPavlov/rubert-base-cased"
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    logger.info(f"Using model: {model_name}")
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512
        )
    logger.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(model_save_path, "logs"),
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        learning_rate=learning_rate,
        save_total_limit=2,
        metric_for_best_model="eval_loss"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    logger.info("Starting model training...")
    trainer.train()
    
    logger.info(f"Saving model to {model_save_path}...")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    metrics = trainer.evaluate()
    with open(os.path.join(model_save_path, "metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)

    dataset_info = {
        "num_examples": len(texts),
        "positive_examples": sum(1 for score in scores if score > 0),
        "negative_examples": sum(1 for score in scores if score <= 0),
        "avg_positive_score": np.mean([score for score in scores if score > 0]),
        "avg_negative_score": np.mean([score for score in scores if score <= 0]),
        "data_source": data_path if data_path else "default path"
    }
    
    with open(os.path.join(model_save_path, "dataset_info.json"), 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=4)
    
    logger.info("Model training completed!")
    logger.info(f"Final metrics: {metrics}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Russian sentiment analysis model')
    parser.add_argument('--data-path', type=str, default=None,
                      help='Path to the TSV file with training data')
    parser.add_argument('--model-path', type=str, default=None,
                      help='Path to save the trained model')
    parser.add_argument('--epochs', type=int, default=5,
                      help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                      help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                      help='Learning rate')
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data_path,
        model_save_path=args.model_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    ) 