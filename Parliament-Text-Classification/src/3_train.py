import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report


def train_and_evaluate():
    data_path = os.path.join("data", "cleaned", "parliament_petitions_clean.csv")
    print(f"Veri yükleniyor: {data_path}...")
    df = pd.read_csv(data_path)

    # Girdi (X) ve Çıktı (y) değişkenlerini ayır
    X = df['clean_text']
    y = df['label']

    # Veriyi %80 Eğitim, %20 Test olarak böl
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Eğitim seti boyutu: {len(X_train)} | Test seti boyutu: {len(X_test)}\n")

    # Metinleri Matematiksel Vektörlere Çevirme (TF-IDF Feature Extraction)
    print("TF-IDF Vektörizasyonu yapılıyor...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # --- BENCHMARK MODELLERİ ---

    models = {
        "Baseline 1 (Logistic Regression)": LogisticRegression(max_iter=1000, random_state=42),
        "Baseline 2 (Linear SVC)": LinearSVC(random_state=42, dual=False)
    }

    results = []

    print("\n" + "=" * 50)
    print("MODEL EĞİTİMİ VE BENCHMARK BAŞLIYOR")
    print("=" * 50)

    for model_name, model in models.items():
        print(f"\n[{model_name}] eğitiliyor...")
        model.fit(X_train_vec, y_train)

        # Test seti üzerinde tahmin yap
        y_pred = model.predict(X_test_vec)

        # Metrikleri hesapla
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        results.append({"Model": model_name, "Accuracy": acc, "Macro_F1": macro_f1})

        print(f"{model_name} Sonuçları:")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Macro F1:  {macro_f1:.4f}")

        # (Opsiyonel) Hocanın sunumda sorabileceği detaylı rapor
        # print(classification_report(y_test, y_pred))

    # Benchmark Tablosu
    print("\n" + "=" * 50)
    print("NİHAİ BENCHMARK TABLOSU")
    print("=" * 50)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("=" * 50)

    print("\nİpucu: Sunumda Macro F1 skorlarını kıyaslamayı unutmayın!")


if __name__ == "__main__":
    train_and_evaluate()