import numpy as np
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import torch

class fineTuner:
    def __init__(self, **kwargs):
        self.model_name = kwargs.get('model_name','distilbert/distilbert-base-uncased')
        self.max_token_length = kwargs.get('max_token_length',128)
        self.metrics_list = kwargs.get('metrics_list', ['accuracy', 'precision', 'recall', 'f1'])
        self.output_dir = kwargs.get('output_dir','./output')
        self.num_train_epochs = kwargs.get('num_train_epochs',2)
        self.per_device_train_batch_size = kwargs.get('per_device_train_batch_size',8)
        self.per_device_eval_batch_size = kwargs.get('per_device_eval_batch_size',8)
        self.learning_rate = kwargs.get('learning_rate',5e-5)
        self.weight_decay = kwargs.get('weight_decay',0.0)
        self.logging_strategy = kwargs.get('logging_strategy','epoch')
        self.report_to = kwargs.get('report_to','none')

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.dataset = None
        self.model = None
        self.training_args = None
        self.trainer = None
        self.metrics = None
        self.is_trained = False

        self.device = torch.device("cpu")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")


    def tokenize(self, input):
        return self.tokenizer(
            input["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_token_length
        )
        

    def create_dataset(self, training_data, validation_data, testing_data, label_column = 'label', text_column = 'text'):
        self.dataset = DatasetDict({
            'train': Dataset.from_pandas(training_data[[label_column, text_column]].reset_index(drop=True)),
            'validation': Dataset.from_pandas(validation_data[[label_column, text_column]].reset_index(drop=True)),
            'test': Dataset.from_pandas(testing_data[[label_column, text_column]].reset_index(drop=True))
        })

        
        self.tokenized_dataset = self.dataset.map(self.tokenize, batched=True)

    def define_trainer(self):
        if (self.dataset is None):
            print('ERROR: Please define your dataset using create_dataset() first.')
            return

        NUM_LABELS = len(np.unique(self.dataset['train']['label']))

        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=NUM_LABELS)
        self.metrics = evaluate.combine(self.metrics_list)

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return self.metrics.compute(predictions=predictions, references=labels)
        
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            logging_strategy=self.logging_strategy,
            report_to=self.report_to
        )

        # account for different number of elements per label
        # (might not always be necessary, e.g., if training data is already balanced)
        ulabels = np.unique(self.dataset['train']['label']) 
        label_counts = [self.dataset['train']['label'].count(lbl) for lbl in ulabels]
        total = sum(label_counts)
        weights = [total/c for c in label_counts]  
        label_weights = torch.tensor(weights, dtype=torch.float).to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=label_weights)
        
        def compute_loss_func(outputs, labels, **kwargs):
            logits = outputs.logits
            return loss_fn(logits, labels)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["validation"],
            compute_metrics=compute_metrics,
            compute_loss_func=compute_loss_func
        )

        self.is_trained = False

    def train_and_evaluate(self):
        if (self.is_trained):
            print('ERROR: This model is already fine tuned.  Please redefine the model with define_trainer() first.')
            return None, None
        training_output = self.trainer.train()
        evaluate_output = self.trainer.evaluate(eval_dataset=self.tokenized_dataset["test"])
        self.is_trained = True
        return training_output, evaluate_output
    
    def create_finetuned_model_from_data(self, training_data, validation_data, testing_data, label_column = 'label', text_column = 'text'):
        self.create_dataset(training_data, validation_data, testing_data, label_column = label_column, text_column = text_column)
        self.define_trainer()
        training_output, evaluate_output = self.train_and_evaluate()
        return training_output, evaluate_output
    
    def predict(self, new_data_df, text_column = 'text'):
        if (self.dataset is None):
            print('ERROR: Please define your trainer using create_dataset(), define_trainer() and train_and_evaluate() first.')
            return None, None
        
        new_dataset = Dataset.from_pandas(new_data_df[[text_column]].reset_index(drop=True))
        new_dataset_tokenized = new_dataset.map(self.tokenize, batched=True)
        predictions = self.trainer.predict(new_dataset_tokenized)
        predicted_classes = np.argmax(predictions.predictions, axis=1)

        return predictions, predicted_classes
    










