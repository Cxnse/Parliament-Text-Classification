import pandas as pd
import re
import os


def clean_text(text):
    """Metinleri NLP modelleri için Regex ile temizler."""
    if not isinstance(text, str):
        return ""

    # 1. Her şeyi küçük harfe çevir
    text = text.lower()

    # 2. Regex: Sadece İngilizce harfleri, sayıları ve boşlukları tut;
    # Noktalama işaretlerini, özel karakterleri ve emojileri sil.
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # 3. Regex: Yan yana gelmiş birden fazla boşluğu veya alt satırı tek boşluğa indir.
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_data(input_path, output_path, top_n_classes=5):
    print(f"Veri yükleniyor: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Orijinal veri boyutu: {df.shape}\n")

    # Eksik/Boş (NaN) verileri sil
    df.dropna(subset=['text', 'label'], inplace=True)

    print("NLP Metin Temizleme (Regex) işlemi uygulanıyor, lütfen bekleyin...")
    df['clean_text'] = df['text'].apply(clean_text)

    # Çok kısa olan (10 karakterden az, yani anlamsız) metinleri çıkar
    df = df[df['clean_text'].str.len() > 10]

    print("\n--- Filtreleme Öncesi İlk 10 Bakanlık Dağılımı ---")
    print(df['label'].value_counts().head(10))

    # SINIF DENGESİZLİĞİ (Class Imbalance) ÇÖZÜMÜ:
    # Sadece en çok dilekçe alan ilk N sınıfı tutalım ki model öğrenebilsin.
    top_departments = df['label'].value_counts().nlargest(top_n_classes).index
    df_filtered = df[df['label'].isin(top_departments)]

    print(f"\n--- Sadece En Popüler {top_n_classes} Bakanlık (Sınıf) Seçildi ---")
    print(f"Filtrelenmiş ve Temizlenmiş veri boyutu: {df_filtered.shape}")
    print(df_filtered['label'].value_counts())

    # Sadece modele girecek 'clean_text' ve 'label' sütunlarını kaydedelim
    df_final = df_filtered[['clean_text', 'label']]
    df_final.to_csv(output_path, index=False)
    print(f"\nİşlem Tamam! Temiz veri '{output_path}' olarak kaydedildi.")


if __name__ == "__main__":
    # Dosya yolları (os.path.join ile işletim sistemi bağımsız)
    RAW_DATA_PATH = os.path.join("data", "raw", "parliament_petitions_raw.csv")
    CLEAN_DATA_PATH = os.path.join("data", "cleaned", "parliament_petitions_clean.csv")

    preprocess_data(RAW_DATA_PATH, CLEAN_DATA_PATH, top_n_classes=5)