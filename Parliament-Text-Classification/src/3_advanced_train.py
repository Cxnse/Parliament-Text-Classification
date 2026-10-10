import pandas as pd
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import pipeline
from datasets import Dataset


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    return {'accuracy': acc, 'macro_f1': macro_f1}


def advanced_pipeline():
    # 1. Veri Hazırlığı
    data_path = os.path.join("data", "cleaned", "parliament_petitions_clean.csv")
    df = pd.read_csv(data_path)

    # Etiketleri (Label) sayılara (0, 1, 2, 3, 4) çevir
    labels = df['label'].unique().tolist()
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    df['label_id'] = df['label'].map(label2id)

    X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label_id'], test_size=0.2,
                                                        random_state=42, stratify=df['label_id'])

    # 2. HOCANIN İSTEDİĞİ VANILLA BASELINE: BERT-base-uncased
    print("\n--- 1. AŞAMA: VANILLA BASELINE (BERT-base-uncased) ---")
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Veriyi HuggingFace formatına çevir
    train_dataset = Dataset.from_dict({'text': X_train.tolist(), 'label': y_train.tolist()})
    test_dataset = Dataset.from_dict({'text': X_test.tolist(), 'label': y_test.tolist()})

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # M2 Çip (MPS) kontrolü
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Kullanılan Donanım: {device}")

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels))
    model.to(device)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    print("BERT Eğitimi Başlıyor (M2 gücü devrede)...")
    trainer.train()
    bert_results = trainer.evaluate()

    # 3. JEV ALTERNATİFİ (Zero-Shot LLM-as-a-judge)
    print("\n--- 2. AŞAMA: JEV ALTERNATİFİ (Zero-Shot LLM Decision) ---")
    print("Zero-Shot modeli indiriliyor (Ücretsiz LLM Hakem)...")

    # JEV için ücretsiz, güçlü bir zero-shot LLM kullanıyoruz
    jev_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli",
                            device=0 if str(device) == "mps" else -1)

    print("LLM-as-a-judge (JEV) test seti üzerinde kararlar veriyor (Zaman alabilir)...")

    # Süreyi kısa tutmak için test setinden rastgele 50 örnek üzerinde JEV simülasyonu
    sample_texts = X_test.tolist()[:50]
    true_labels = y_test.tolist()[:50]
    jev_preds = []

    for i, text in enumerate(sample_texts):
        if i % 10 == 0:
            print(f"JEV İlerlemesi: {i}/50")
        # LLM'e soruyoruz: "Sence bu metin bu 5 bakanlıktan hangisine ait?"
        result = jev_pipeline(text, candidate_labels=labels)
        predicted_label = result['labels'][0]
        jev_preds.append(label2id[predicted_label])

    jev_acc = accuracy_score(true_labels, jev_preds)
    jev_macro_f1 = f1_score(true_labels, jev_preds, average='macro')

    print("\n" + "=" * 50)
    print("NİHAİ BENCHMARK TABLOSU (HOCANIN İSTEDİĞİ FORMAT)")
    print("=" * 50)
    print(f"1. Vanilla Baseline (BERT-base-uncased) - Macro F1: {bert_results['eval_macro_f1']:.4f}")
    print(f"2. Geleneksel Baseline (Linear SVC)       - Macro F1: 0.8768 (Önceki testten)")
    print(f"3. JEV Alternatifi (Zero-Shot LLM)        - Macro F1: {jev_macro_f1:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    advanced_pipeline()