import requests
import pandas as pd
import time
import os


def fetch_petitions(target_count=3000):
    """
    İngiltere Parlamentosu API'sinden dilekçeleri çeker.
    """
    base_url = "https://petition.parliament.uk/petitions.json"
    petitions_data = []
    current_url = f"{base_url}?state=all"

    print("Dilekçeler çekiliyor, lütfen bekleyin...")

    while current_url and len(petitions_data) < target_count:
        response = requests.get(current_url)

        if response.status_code != 200:
            print("API Bağlantı Hatası:", response.status_code)
            break

        data = response.json()

        for item in data.get("data", []):
            attr = item["attributes"]
            # Sadece bir bakanlığa atanmış dilekçeleri alıyoruz (Etiket/Label için şart)
            if attr.get("departments"):
                # Dilekçenin başlığı ve arka plan detayını birleştiriyoruz (Özellik/Feature)
                full_text = str(attr.get("action", "")) + " " + str(attr.get("background", ""))
                # Hangi bakanlığa atanmış (Sınıf/Label)
                department = attr["departments"][0]["name"]

                petitions_data.append({
                    "text": full_text,
                    "label": department
                })

        # Sonraki sayfaya geç (eğer varsa)
        next_page = data.get("links", {}).get("next")
        current_url = next_page if next_page else None

        # API'yi yormamak için ufak bir bekleme
        time.sleep(1)
        print(f"Şu ana kadar çekilen dilekçe sayısı: {len(petitions_data)}")

    # Veriyi Pandas DataFrame'e çevirip kaydediyoruz
    df = pd.DataFrame(petitions_data)

    # Dosya yolu (data/raw klasörüne kaydedecek)
    output_path = os.path.join("data", "raw", "parliament_petitions_raw.csv")
    df.to_csv(output_path, index=False)
    print(f"\nİşlem Tamam! {len(df)} adet veri '{output_path}' olarak kaydedildi.")
    return df


if __name__ == "__main__":
    df_raw = fetch_petitions(target_count=3000)