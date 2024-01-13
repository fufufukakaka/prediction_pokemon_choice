import cloudpickle
import torch
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)


def main():
    tokenizer = AutoTokenizer.from_pretrained("fufufukakaka/pokemon_team_BERT")
    model = BertForSequenceClassification.from_pretrained("fufufukakaka/pokemon_team_BERT", num_labels=3)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    def label_to_number(example):
        label_dict = {'first_choiced': 0, 'choiced': 1, 'not_choiced': 2}
        example['label'] = label_dict[example['label']]
        return example


    data = cloudpickle.load(open("data/pokemon_selection/poke_battle_logger_pokemon_selection_preprocessed.pkl", "rb"))
    train, test = train_test_split(data, test_size=0.1, random_state=42)
    train, validation = train_test_split(train, test_size=0.1, random_state=42)

    dataset = Dataset.from_list(train)
    train_tokenized_dataset = dataset.map(tokenize_function, batched=True)
    train_tokenized_dataset = train_tokenized_dataset.map(label_to_number)
    dataset = Dataset.from_list(validation)
    validation_tokenized_dataset = dataset.map(tokenize_function, batched=True)
    validation_tokenized_dataset = validation_tokenized_dataset.map(label_to_number)
    # dataset = Dataset.from_list(test)
    # test_tokenized_dataset = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=10,
        weight_decay=0.01,
        save_total_limit = 5,
    )

    # トレーナーの初期化
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_dataset,
        eval_dataset=validation_tokenized_dataset
    )

    # トレーニングの実行
    trainer.train()


if __name__ == "__main__":
    main()
