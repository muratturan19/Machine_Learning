# 📦 Gerekli kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
print(f"Kullanılan SHAP Kütüphane Sürümü: {shap.__version__}")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, classification_report,
    precision_recall_curve, roc_auc_score, roc_curve
)
from sklearn.inspection import PartialDependenceDisplay

# 🔹 Google Colab'da dosya yükleme
from google.colab import files
print("Lütfen 'simule_uretim_verisi.csv' adlı veri dosyanızı seçin:")
uploaded = files.upload()

# 🔹 Veriyi oku
df = None
df_original_for_ts_plots = None # Zaman serisi çizimleri için orijinal df'i sakla
try:
    file_name_to_load = "simule_uretim_verisi.csv" # Yüklenecek dosya adı bu olmalı
    actual_file_name_uploaded = None

    if len(uploaded.keys()) == 0:
        raise FileNotFoundError("Hiçbir dosya yüklenmedi.")
    
    # Yüklenen dosya adını bul (Colab bazen dosya adını değiştirebilir)
    # En olası eşleşmeyi bulmaya çalışalım
    if file_name_to_load in uploaded:
        actual_file_name_uploaded = file_name_to_load
    else: # Eğer tam eşleşme yoksa, yüklenen ilk dosyayı al
        actual_file_name_uploaded = list(uploaded.keys())[0]
        print(f"UYARI: Beklenen dosya adı '{file_name_to_load}' bulunamadı. Yüklenen dosya: '{actual_file_name_uploaded}' kullanılacak.")
            
    df = pd.read_csv(io.BytesIO(uploaded[actual_file_name_uploaded])) # io.BytesIO eklendi
    print(f"'{actual_file_name_uploaded}' başarıyla yüklendi.")
    print(f"Veri seti boyutu: {df.shape}")
    print("\nVeri Setinin İlk 5 Satırı:")
    print(df.head())
    print("\nVeri Seti Bilgileri:")
    df.info()
    df_original_for_ts_plots = df.copy() # Orijinal df'i kopyala

except FileNotFoundError as fnf_e:
    print(f"HATA: Dosya yüklenmedi veya bulunamadı. {fnf_e}")
    exit()
except Exception as e:
    print(f"Dosya okunurken bir hata oluştu: {e}")
    exit()

# 📌 Korelasyon Matrisi (HAM VERİ ÜZERİNDEN - Tarih ve kategorikler çıkarılmadan önce)
try:
    print("\n--- Ham Veri Korelasyon Matrisi ---")
    plt.figure(figsize=(12, 10))
    # Sadece sayısal sütunları al, object/string tipindekileri ve tarihi dışarıda bırakmaya çalış
    numeric_cols_for_raw_corr = df.select_dtypes(include=np.number).columns.tolist()
    if 'tarih' in df.columns and df['tarih'].dtype == 'object': # Eğer tarih object ise korelasyondan çıkar
        pass # Zaten select_dtypes(include=np.number) bunu yapacaktır
    
    if numeric_cols_for_raw_corr:
        sns.heatmap(df[numeric_cols_for_raw_corr].corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Korelasyon Matrisi (Sadece Sayısal Özellikler - Ham Veri)")
        plt.tight_layout()
        plt.show()
    else:
        print("Ham veri korelasyon matrisi için sayısal sütun bulunamadı.")
except Exception as e:
    print(f"İlk korelasyon matrisi çizilirken hata: {e}")


# --- ÖZELLİK MÜHENDİSLİĞİ: TARİH VE OLAY BİLGİLERİ ---
df_processed = df.copy() 
newly_created_time_features = []
# Kategorik ve sayısal özellikleri ham df_processed üzerinden belirleyelim
temp_X_for_feature_detection = df_processed.drop('hata_var', axis=1, errors='ignore')

# 'vardiya' ve 'makine_id' kategorik olarak kabul edilecekse
original_categorical_features = []
if 'vardiya' in temp_X_for_feature_detection.columns: original_categorical_features.append('vardiya')
if 'makine_id' in temp_X_for_feature_detection.columns: original_categorical_features.append('makine_id') # makine_id'yi de kategorik alalım

original_numerical_features = [
    col for col in temp_X_for_feature_detection.columns 
    if col not in original_categorical_features and col != 'tarih' and \
    df_processed[col].dtype in [np.int64, np.int32, np.float64, int, float]
]

try:
    print("\n🔧 Özellik Mühendisliği Başlıyor...")
    if 'tarih' in df_processed.columns:
        df_processed['tarih'] = pd.to_datetime(df_processed['tarih'])
        df_processed['yil'] = df_processed['tarih'].dt.year
        df_processed['ay'] = df_processed['tarih'].dt.month
        df_processed['gun'] = df_processed['tarih'].dt.day
        df_processed['haftanin_gunu'] = df_processed['tarih'].dt.dayofweek # Pazartesi=0, Pazar=6
        df_processed['yilin_gunu'] = df_processed['tarih'].dt.dayofyear
        df_processed['hafta_numarasi'] = df_processed['tarih'].dt.isocalendar().week.astype(int)
        newly_created_time_features = ['yil', 'ay', 'gun', 'haftanin_gunu', 'yilin_gunu', 'hafta_numarasi']
        print(f"'tarih' sütunundan yeni zaman özellikleri türetildi: {newly_created_time_features}")
        
        # === YENİ: Zaman Serisi Grafikleri için Tarih Index'li df_original_for_ts_plots'u kullanalım ===
        if df_original_for_ts_plots is not None and 'tarih' in df_original_for_ts_plots.columns:
            try:
                df_original_for_ts_plots['tarih'] = pd.to_datetime(df_original_for_ts_plots['tarih'])
                df_original_for_ts_plots.set_index('tarih', inplace=True)
                print("Orijinal veri seti zaman serisi analizleri için indexlendi.")
            except Exception as e_ts_index:
                print(f"Zaman serisi için orijinal df indexlenirken hata: {e_ts_index}")
        
        if 'rulman_degisimi_sonrasi' in df_processed.columns:
            print("'rulman_degisimi_sonrasi' özelliği zaten mevcut.")
            if df_processed['rulman_degisimi_sonrasi'].dtype == 'object':
                try: df_processed['rulman_degisimi_sonrasi'] = pd.to_numeric(df_processed['rulman_degisimi_sonrasi'], errors='coerce').fillna(0) # Sayıya çevir, çeviremezse 0 yap
                except ValueError: print(f"UYARI: 'rulman_degisimi_sonrasi' sayıya çevrilemedi.")
        
        # 'tarih' sütunu, zaman özellikleri türetildikten sonra modelleme için çıkarılır
        df_processed = df_processed.drop('tarih', axis=1)
        print("'tarih' sütunu modelleme için özelliklerden çıkarıldı (zaman özellikleri türetildi).")
    else:
        print("UYARI: 'tarih' sütunu bulunamadı. Zaman özellikleri türetilemedi.")
except Exception as e:
    print(f"Özellik mühendisliği sırasında hata oluştu: {e}")
    df_processed = df.copy() 
    if 'tarih' in df_processed.columns:
        df_processed = df_processed.drop('tarih', axis=1, errors='ignore')

# === YENİ: ZAMAN SERİSİ GÖRSELLEŞTİRMELERİ (ÖZELLİK MÜHENDİSLİĞİNDEN SONRA, df_original_for_ts_plots ÜZERİNDEN) ===
if df_original_for_ts_plots is not None and isinstance(df_original_for_ts_plots.index, pd.DatetimeIndex):
    print("\n--- Zaman Serisi Görselleştirmeleri (HAM VERİ ÜZERİNDEN) ---")

    # Tek bir makinenin titreşim verisi (Örnek: ilk makine_id)
    if 'makine_id' in df_original_for_ts_plots.columns and 'titreşim' in df_original_for_ts_plots.columns:
        makine_id_to_plot = df_original_for_ts_plots['makine_id'].unique()[0]
        makine_data = df_original_for_ts_plots[df_original_for_ts_plots['makine_id'] == makine_id_to_plot]['titreşim']
        if not makine_data.empty:
            plt.figure(figsize=(15, 5))
            makine_data.plot(title=f'Makine {makine_id_to_plot} - Titreşim Zaman Serisi', legend=True)
            plt.ylabel('Titreşim')
            plt.show()

    # Tüm makinelerin ortalama sıcaklık trendi (Günlük ortalama)
    if 'sicaklik' in df_original_for_ts_plots.columns:
        plt.figure(figsize=(15, 5))
        df_original_for_ts_plots['sicaklik'].resample('D').mean().plot(title='Günlük Ortalama Sıcaklık Zaman Serisi', legend=True)
        plt.ylabel('Ortalama Sıcaklık [K]')
        plt.show()

    # Rulman_degisimi_sonrasi etkisini incele (Tüm makine ID'leri için genel veya örnek bir makine)
    if 'rulman_degisimi_sonrasi' in df_original_for_ts_plots.columns and 'titreşim' in df_original_for_ts_plots.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='rulman_degisimi_sonrasi', y='titreşim', data=df_original_for_ts_plots)
        plt.title('Rulman Durumuna Göre Titreşim Dağılımı (Tüm Makineler)')
        plt.show()

        # Örnek bir makine için zaman serisinde rulman durumunun etkisi
        makine_id_for_rulman_plot = df_original_for_ts_plots['makine_id'].unique()[0] # İlk makine
        df_makine_rulman = df_original_for_ts_plots[df_original_for_ts_plots['makine_id'] == makine_id_for_rulman_plot]
        if not df_makine_rulman.empty:
            plt.figure(figsize=(15, 7))
            sns.lineplot(data=df_makine_rulman, x=df_makine_rulman.index, y='titreşim', hue='rulman_degisimi_sonrasi', palette='viridis', legend='full')
            plt.title(f'Makine {makine_id_for_rulman_plot} - Titreşim ve Rulman Değişimi Sonrası Durum')
            plt.ylabel('Titreşim')
            plt.show()
else:
    print("\nZaman serisi grafikleri için 'tarih' index olarak ayarlanmalı veya df_original_for_ts_plots uygun değil.")

# === YENİ: HATA DURUMLARI VE RULMAN İLİŞKİSİ (df_processed üzerinden, özellikler türetildikten sonra) ===
if 'hata_var' in df_processed.columns and 'rulman_degisimi_sonrasi' in df_processed.columns:
    print("\n--- Hata Durumları ve Rulman İlişkisi ---")
    hata_rulman_crosstab = pd.crosstab(df_processed['rulman_degisimi_sonrasi'], df_processed['hata_var'])
    print("\nRulman Durumuna Göre Hata Sayıları:\n", hata_rulman_crosstab)
    if not hata_rulman_crosstab.empty:
        hata_rulman_crosstab.plot(kind='bar', stacked=False, figsize=(12,7), colormap='viridis')
        plt.title('Rulman Durumuna Göre Hata Dağılımı')
        plt.ylabel('Sayı')
        plt.xlabel('Rulman Değişimi Sonrası Durum (0-4)')
        plt.xticks(rotation=0)
        plt.legend(title='Hata Var', labels=['Hata Yok', 'Hata Var'])
        plt.show()

# === PROPHET İLE ZAMAN SERİSİ TAHMİNİ (YENİ BÖLÜM) ===
print("\n--- Prophet ile Zaman Serisi Tahmini ---")
try:
    from prophet import Prophet
    print("Prophet kütüphanesi başarıyla yüklendi.")

    if df_original_for_ts_plots is not None and isinstance(df_original_for_ts_plots.index, pd.DatetimeIndex):
        # Örnek olarak ilk makinenin titreşim verisini alalım
        makine_id_for_prophet = df_original_for_ts_plots['makine_id'].unique()[0]
        df_prophet_input = df_original_for_ts_plots[df_original_for_ts_plots['makine_id'] == makine_id_for_prophet][['titreşim']].reset_index()
        df_prophet_input.rename(columns={'tarih': 'ds', 'titreşim': 'y'}, inplace=True)

        if not df_prophet_input.empty and len(df_prophet_input) > 2 : # Prophet en az 2 veri noktası ister
            print(f"\nMakine {makine_id_for_prophet} için Prophet modeli hazırlanıyor (hedef: titreşim)...")

            # Olayları (rulman değişimi) tanımla
            # 'rulman_degisimi_sonrasi == 0' olan tarihleri bulalım
            # Bu kısım, rulman_degisimi_sonrasi sütununun anlamına göre uyarlanmalı
            rulman_degisim_tarihleri = df_original_for_ts_plots[
                (df_original_for_ts_plots['makine_id'] == makine_id_for_prophet) & 
                (df_original_for_ts_plots['rulman_degisimi_sonrasi'] == 0) # Veya değişimi gösteren başka bir mantık
            ].index.tolist()

            holidays_df = None
            if rulman_degisim_tarihleri:
                holidays_df = pd.DataFrame({
                    'holiday': 'rulman_degisimi',
                    'ds': pd.to_datetime(rulman_degisim_tarihleri),
                    'lower_window': 0, # Olayın sadece o gün etkili olduğunu varsayalım
                    'upper_window': 1  # Olayın ertesi gün de bir miktar etkisi olabilir (ayarlanabilir)
                })
                print(f"Rulman değişimi olayları bulundu: {len(holidays_df)} adet")

            # Prophet modelini oluştur ve eğit
            prophet_model = Prophet(holidays=holidays_df, daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True, seasonality_mode='additive')
            # Gerekirse 'changepoint_prior_scale' gibi parametreler ayarlanabilir.
            
            # Eğer harici regresör (extra regressors) eklemek istersek:
            # Örneğin, üretim hızının titreşimi etkilediğini düşünüyorsak:
            # df_prophet_input['uretim_hizi'] = df_original_for_ts_plots.loc[df_prophet_input['ds'], 'uretim_hizi'].values # ds'e göre eşleştir
            # prophet_model.add_regressor('uretim_hizi')

            prophet_model.fit(df_prophet_input)

            # Gelecek için tahmin yap (örneğin 90 gün sonrası)
            future_dates = prophet_model.make_future_dataframe(periods=90, freq='D')
            
            # Eğer regresör eklediysek future_dates'e o regresörün gelecek değerlerini de eklememiz gerekir.
            # if 'uretim_hizi' in df_prophet_input.columns:
            #    # Bu kısım için üretim hızının gelecek değerlerini tahmin etmek veya varsaymak gerekir.
            #    # Basitlik adına son değeri sabit tutabiliriz veya bir trend uygulayabiliriz.
            #    last_uretim_hizi = df_prophet_input['uretim_hizi'].iloc[-1]
            #    future_dates['uretim_hizi'] = last_uretim_hizi # Çok basit bir varsayım

            forecast = prophet_model.predict(future_dates)

            print("\nTahmin Sonuçları (İlk Birkaçı ve Son Birkaçı):")
            print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
            print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

            # Tahminleri ve bileşenleri çizdir
            fig1 = prophet_model.plot(forecast)
            plt.title(f'Makine {makine_id_for_prophet} - Titreşim Tahmini')
            plt.show()

            fig2 = prophet_model.plot_components(forecast)
            plt.show()
            
        else:
            print(f"Makine {makine_id_for_prophet} için Prophet modeli eğitಲು yeterli veri yok.")
    else:
        print("Prophet analizi için zaman indexli orijinal veri seti (df_original_for_ts_plots) bulunamadı.")

except ImportError:
    print("Prophet kütüphanesi bulunamadı. Yüklemek için: !pip install prophet")
except Exception as e_prophet:
    print(f"Prophet analizi sırasında bir hata oluştu: {e_prophet}")


# X ve y'yi özellik mühendisliği tamamlanmış df_processed'dan ayıralım
if 'hata_var' not in df_processed.columns:
    print("HATA: Hedef değişken 'hata_var' df_processed içinde bulunamadı!")
    exit()

X = df_processed.drop('hata_var', axis=1) 
y = df_processed['hata_var']

# Kategorik ve sayısal özellikleri X üzerinden yeniden belirleyelim (türetilmiş zaman özellikleri dahil)
categorical_features = []
if 'vardiya' in X.columns: categorical_features.append('vardiya') # One-hot sonrası bu olmayacak
if 'makine_id' in X.columns: categorical_features.append('makine_id') # Kategorik alınacaksa

# One-Hot sonrası oluşan 'vardiya_...' sütunlarını yakala
one_hot_vardiya_cols = [col for col in X.columns if col.startswith('vardiya_')]
# 'yil', 'ay' gibi türetilmiş zaman özellikleri de sayısal ama farklı ele alınabilir
# Şimdilik tüm sayısal görünenleri alalım

# Nihai kategorik özellikler (One-Hot öncesi)
final_categorical_to_encode = []
if 'vardiya' in X.columns: final_categorical_to_encode.append('vardiya') # 'vardiya' OneHot ile işlenecekse
if 'makine_id' in df.columns and df['makine_id'].nunique() < 20 : # Eğer makine_id çok çeşitli değilse kategorik alınabilir.
    final_categorical_to_encode.append('makine_id')


final_numerical_features = [
    col for col in X.columns 
    if col not in final_categorical_to_encode and col not in newly_created_time_features and \
    X[col].dtype in [np.int64, np.int32, np.float64]
]
# Zaman özelliklerini de sayısal olarak ekleyebiliriz
final_numerical_features.extend(newly_created_time_features)
final_numerical_features = list(set(final_numerical_features)) # Benzersiz yap

print(f"\nÖzellik mühendisliği sonrası X'in sütunları: {X.columns.tolist()}")
print(f"Model için kullanılacak Sayısal Özellikler: {final_numerical_features}")
print(f"Model için kullanılacak Kategorik Özellikler (One-Hot öncesi): {final_categorical_to_encode}")


# Eksik veri doldurma (Eğer özellik mühendisliği sonrası oluştuysa)
for col in final_numerical_features:
    if col in X.columns and X[col].isnull().sum() > 0: X[col].fillna(X[col].mean(), inplace=True)
for col in final_categorical_to_encode: # One-hot öncesi kategorik sütunlar için
    if col in X.columns and X[col].isnull().sum() > 0: X[col].fillna(X[col].mode()[0], inplace=True)
print("Eksik veriler (gerekliyse) dolduruldu.")


# Preprocessing pipeline
transformers_list = []
# Sayısal özellikler
if valid_numerical_features := [f for f in final_numerical_features if f in X.columns]:
    transformers_list.append(('num', StandardScaler(), valid_numerical_features))
# Kategorik özellikler (One-Hot)
if valid_categorical_features := [f for f in final_categorical_to_encode if f in X.columns]:
    transformers_list.append(('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), valid_categorical_features))

if not transformers_list: print("HATA: Ön işleme için özellik bulunamadı."); exit()
preprocessor = ColumnTransformer(transformers=transformers_list, remainder='passthrough') # remainder='passthrough' önemli


# Veriyi Eğitim ve Test Setlerine Ayırma
try: 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
except ValueError as e: 
    print(f"Stratify hatası: {e}. Stratify olmadan devam ediliyor.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"Eğitim seti: {X_train.shape}, Test seti: {X_test.shape}")


# Modeller (Mevcut kodunuzdaki gibi)
models_to_evaluate = {
    "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced', max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
    "Support Vector Machine": SVC(probability=True, class_weight='balanced', random_state=42)
}
xgboost_available = False; lightgbm_available = False
try:
    from xgboost import XGBClassifier
    count_class_0_train = sum(y_train == 0); count_class_1_train = sum(y_train == 1)
    scale_pos_weight_xgb_train = count_class_0_train / count_class_1_train if count_class_1_train > 0 else 1
    models_to_evaluate["XGBoost"] = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight_xgb_train)
    xgboost_available = True
except ImportError: print_once("XGBoost kütüphanesi bulunamadı, atlanıyor.")
try:
    from lightgbm import LGBMClassifier
    models_to_evaluate["LightGBM"] = LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=-1) # is_unbalance=True de denenebilir
    lightgbm_available = True
except ImportError: print_once("LightGBM kütüphanesi bulunamadı, atlanıyor.")

# Model Değerlendirme Döngüsü (Mevcut kodunuzdaki gibi)
model_results = {}
fig_roc_main, ax_roc_main = plt.subplots(figsize=(10, 8)); ax_roc_main.plot([0, 1], [0, 1], 'k--')

for model_name, classifier_instance in models_to_evaluate.items():
    print(f"\n--- {model_name} Modeli ---")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier_instance)])
    try:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba_positive_class = pipeline.predict_proba(X_test)[:, 1] if hasattr(classifier_instance, "predict_proba") else None
        
        accuracy = accuracy_score(y_test, y_pred)
        f1_hata_var = f1_score(y_test, y_pred, pos_label=1, zero_division=0) # Hata sınıfı (1) için F1
        report_dict = classification_report(y_test, y_pred, target_names=['Hata Yok (0)', 'Hata Var (1)'], output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba_positive_class) if y_proba_positive_class is not None else np.nan
        
        model_results[model_name] = {
            "Pipeline": pipeline, "Accuracy": accuracy, "F1 Score (Hata Var)": f1_hata_var,
            "Precision (Hata Var)": report_dict["Hata Var (1)"]["precision"], 
            "Recall (Hata Var)": report_dict["Hata Var (1)"]["recall"],
            "F1 Macro": report_dict["macro avg"]["f1-score"], "ROC AUC": roc_auc, 
            "Confusion Matrix": cm,
            "Probabilities (Pozitif Sınıf)": y_proba_positive_class
        }
        
        print(f"Accuracy: {accuracy:.4f}, F1 (Hata Var): {f1_hata_var:.4f}, ROC AUC: {roc_auc:.4f}" if y_proba_positive_class is not None else f"Acc: {accuracy:.4f}, F1: {f1_hata_var:.4f}, ROC AUC: N/A")
        print(classification_report(y_test, y_pred, target_names=['Hata Yok (0)', 'Hata Var (1)'], zero_division=0))
        
        fig_cm_ind, ax_cm_ind = plt.subplots(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm_ind);
        ax_cm_ind.set_title(f"Confusion Matrix - {model_name}"); ax_cm_ind.set_xlabel("Tahmin Edilen"); ax_cm_ind.set_ylabel("Gerçek"); plt.show()
        
        if y_proba_positive_class is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba_positive_class); 
            ax_roc_main.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
            
    except Exception as e: print(f"{model_name} ile hata: {type(e).__name__}: {e}")

ax_roc_main.set_xlabel('False Positive Rate'); ax_roc_main.set_ylabel('True Positive Rate'); ax_roc_main.set_title('Modeller için ROC Eğrileri'); ax_roc_main.legend(loc='lower right'); ax_roc_main.grid(True); plt.show()

# Sonuçları DataFrame'e Aktarma (Mevcut kodunuzdaki gibi)
results_list_for_df = [(name, res.get("Accuracy",np.nan), res.get("F1 Score (Hata Var)",np.nan), res.get("Precision (Hata Var)",np.nan),
                        res.get("Recall (Hata Var)",np.nan), res.get("F1 Macro",np.nan), res.get("ROC AUC",np.nan))
                       for name, res in model_results.items()]
results_df = pd.DataFrame(results_list_for_df, columns=["Model", "Accuracy", "F1 Hata Var", "Precision Hata Var", "Recall Hata Var", "F1 Macro", "ROC AUC"])
print("\n--- Tüm Modellerin Özet Performansı ---"); print(results_df.sort_values(by="F1 Hata Var", ascending=False))


# En İyi Model ve Eşik Ayarlama (Mevcut kodunuzdaki gibi, bazı küçük düzeltmelerle)
best_model_overall_name = None; best_model_overall_pipeline = None; top_n_features_for_detail = 3 # SHAP için
feature_importance_shap_df = pd.DataFrame() 

if not results_df.empty:
    try:
        sorted_results_df = results_df.sort_values(by="F1 Hata Var", ascending=False) # Hata sınıfı F1'e göre sırala
        if not sorted_results_df.empty and sorted_results_df.iloc[0]["F1 Hata Var"] > 0 : # En azından bir miktar F1 skoru varsa
             best_model_overall_name = sorted_results_df.iloc[0]["Model"]
        if best_model_overall_name and best_model_overall_name in model_results: 
            best_model_overall_pipeline = model_results[best_model_overall_name]["Pipeline"]
        if best_model_overall_name: print(f"\nGenel en iyi model (F1 Hata Var'a göre): '{best_model_overall_name}'")
        else: print("En iyi model (F1 Hata var > 0 olan) belirlenemedi.")
    except (IndexError, KeyError) as e: print(f"En iyi model belirlenirken hata: {e}"); best_model_overall_pipeline = None
else: print("Model sonuçları (results_df) boş, en iyi model belirlenemiyor.")


if best_model_overall_pipeline:
    y_proba_best_overall = model_results[best_model_overall_name].get("Probabilities (Pozitif Sınıf)")
    best_threshold_glob = 0.5 
    if y_proba_best_overall is not None:
        print(f"\n--- '{best_model_overall_name}' modeli için Eşik Ayarlama ---")
        precision_th, recall_th, thresholds_pr_th = precision_recall_curve(y_test, y_proba_best_overall)
        if len(precision_th)>1 and len(recall_th)>1 and len(thresholds_pr_th)>0:
            # Precision ve Recall'ın son değeri (recall=0, precision=1) ve eşik dizisinin fazladan elemanını çıkar
            fscore_pr_th=(2*precision_th[:-1]*recall_th[:-1])/(precision_th[:-1]+recall_th[:-1]+1e-9) # 1e-9 sıfıra bölme hatasını önler
            if len(fscore_pr_th)>0:
                ix_pr = np.argmax(fscore_pr_th)
                # thresholds_pr_th, precision_th ve recall_th'den bir eksik elemana sahip olabilir
                best_threshold_glob=thresholds_pr_th[min(ix_pr, len(thresholds_pr_th)-1)] 
                print(f'En İyi Eşik (max F1): {best_threshold_glob:.4f}'); 
                y_pred_best_thresh=(y_proba_best_overall>=best_threshold_glob).astype(int)
                print("\nEşik Ayarlanmış Performans:"); 
                print(f"Acc: {accuracy_score(y_test,y_pred_best_thresh):.4f}, F1 (Hata Var): {f1_score(y_test,y_pred_best_thresh,pos_label=1,zero_division=0):.4f}")
                print(classification_report(y_test,y_pred_best_thresh,target_names=['Hata Yok (0)','Hata Var (1)'],zero_division=0))
                cm_bt = confusion_matrix(y_test,y_pred_best_thresh); 
                plt.figure(figsize=(6,5));sns.heatmap(cm_bt,annot=True,fmt='d',cmap='Greens');plt.title(f"CM - {best_model_overall_name} (Eşik {best_threshold_glob:.2f})");plt.show()
            else: print("F1 skorları eşik ayarlama için hesaplanamadı.")
        else: print("Precision/Recall eğrisi için yeterli veri yok.")
    else: 
        if best_model_overall_name: print(f"'{best_model_overall_name}' için olasılık çıktısı yok, eşik ayarlama atlandı.")
else: print("Eşik ayarlanacak model bulunamadı.")


# SHAP ve PDP Analizleri (Mevcut kodunuzdaki gibi, küçük iyileştirmelerle)
print("\n--- DETAYLI ÖZELLİK ANALİZİ (SHAP VE PDP) ---")
# ... (Mevcut SHAP ve PDP kodunuz buraya gelecek, preprocessor ve classifier'ı pipeline'dan alacak şekilde) ...
# SHAP için X_train_transformed_df_shap ve X_test_transformed_df_shap oluştururken preprocessor'ın doğru kullanıldığından emin olun.
# En iyi modelin (best_model_overall_pipeline) preprocessor ve classifier adımlarını kullanarak bu analizleri yapın.
# SHAP ve PDP kodunuz zaten oldukça detaylı, oradaki mantığı koruyarak, 
# `shap_pipeline_to_analyze` değişkeninin doğru şekilde atandığından emin olun.

# SHAP ve PDP kodunuzun büyük bir kısmını olduğu gibi kullanabilirsiniz,
# Sadece `shap_pipeline_to_analyze`'nin doğru pipeline'ı işaret ettiğinden
# ve `X_train`, `X_test`'in ham (ölçeklenmemiş, encode edilmemiş) versiyonlar olduğundan emin olun,
# çünkü `preprocessor` bu dönüşümleri pipeline içinde yapacak.

# Örnek SHAP/PDP entegrasyonu (kendi kodunuzu buraya uyarlayın):
if shap_pipeline_to_analyze: # Bu değişken yukarıdaki model seçim mantığından gelmeli
    print(f"\n'{shap_model_to_analyze_name}' Üzerinden Detaylı Analizler Başlıyor.")
    try:
        current_preprocessor_shap = shap_pipeline_to_analyze.named_steps['preprocessor']
        classifier_for_shap = shap_pipeline_to_analyze.named_steps['classifier']
        
        # Transformed veriyi al
        X_train_transformed_for_shap = current_preprocessor_shap.transform(X_train)
        X_test_transformed_for_shap = current_preprocessor_shap.transform(X_test)
        
        # get_feature_names_out() ColumnTransformer'dan özellik adlarını almak için
        try:
            all_feature_names_shap = current_preprocessor_shap.get_feature_names_out()
        except AttributeError: # Eski sklearn versiyonları için fallback
            # Bu kısım karmaşık olabilir, en iyisi sklearn'i güncellemek veya manuel adlandırma
            print("UYARI: get_feature_names_out kullanılamıyor. SHAP özellik adları eksik olabilir.")
            all_feature_names_shap = [f"feature_{i}" for i in range(X_train_transformed_for_shap.shape[1])]


        X_train_transformed_df_shap = pd.DataFrame(X_train_transformed_for_shap, columns=all_feature_names_shap)
        X_test_transformed_df_shap = pd.DataFrame(X_test_transformed_for_shap, columns=all_feature_names_shap)

        # ... (Mevcut SHAP explainer ve plot kodlarınız buraya) ...
        # Örnek:
        is_tree_explainer_model = isinstance(classifier_for_shap, (RandomForestClassifier, XGBClassifier if xgboost_available else type(None), LGBMClassifier if lightgbm_available else type(None) ))
        if is_tree_explainer_model:
            explainer = shap.TreeExplainer(classifier_for_shap, X_train_transformed_df_shap, feature_perturbation="interventional")
            shap_values_test = explainer.shap_values(X_test_transformed_df_shap) # Veya sadece explainer(X_test_transformed_df_shap)
            
            # shap_values genellikle (sınıf_sayısı, örnek_sayısı, özellik_sayısı) şeklinde gelir
            # İkili sınıflandırmada pozitif sınıf için olanı alalım:
            shap_values_for_positive_class = shap_values_test[1] if isinstance(shap_values_test, list) and len(shap_values_test) == 2 else shap_values_test

            print("\nSHAP Summary Plot (Beeswarm):")
            shap.summary_plot(shap_values_for_positive_class, X_test_transformed_df_shap, feature_names=all_feature_names_shap, max_display=15)
            
            print("\nSHAP Summary Plot (Bar):")
            shap.summary_plot(shap_values_for_positive_class, X_test_transformed_df_shap, feature_names=all_feature_names_shap, plot_type="bar", max_display=15)

        else:
            print(f"'{shap_model_to_analyze_name}' için uygun SHAP Explainer (Tree) bulunamadı. KernelExplainer veya LinearExplainer deneyebilirsiniz.")

        # PDP
        # ... (Mevcut PDP kodunuz buraya, X_train orijinalini ve shap_pipeline_to_analyze'i kullanarak) ...
        # original_numerical_features_for_binning yerine SHAP'tan gelen en önemli sayısal özellikleri kullanın
        if 'feature_importance_shap_df' in locals() and not feature_importance_shap_df.empty:
            # SHAP'tan gelen özellik adları (örn: num__sicaklik, cat__makine_id_2). Orijinal adlara map etmeniz gerekebilir.
            # Basitlik için, SHAP dataframe'indeki ilk N sayısal özelliği alıp, __ öncesini atarak orijinal adı bulmaya çalışalım.
            top_shap_features_raw = feature_importance_shap_df_V2['feature'].tolist() # V2'yi kendi SHAP df adınızla değiştirin
            pdp_features_to_plot = []
            for f_name_shap in top_shap_features_raw:
                original_f_name = f_name_shap.split('__')[-1] # 'num__sicaklik' -> 'sicaklik'
                if original_f_name in final_numerical_features and original_f_name not in ['yil','ay','gun','haftanin_gunu','yilin_gunu','hafta_numarasi']: # Zaman özelliklerini hariç tutalım
                     if original_f_name not in pdp_features_to_plot:
                        pdp_features_to_plot.append(original_f_name)
                if len(pdp_features_to_plot) >= 3: break # İlk 3 sayısal özelliği al

            if pdp_features_to_plot:
                 print(f"\nPDP için seçilen özellikler: {pdp_features_to_plot}")
                 PartialDependenceDisplay.from_estimator(
                     shap_pipeline_to_analyze, X_train, pdp_features_to_plot, # X_train burada ham, ön işlenmemiş olmalı
                     kind="average", n_jobs=-1, grid_resolution=30
                 )
                 plt.suptitle(f'{shap_model_to_analyze_name} için PDP (Hata Var Olasılığı)', y=1.02, fontsize=16)
                 plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()
            else:
                print("PDP için çizdirilecek uygun sayısal özellik bulunamadı.")
        else:
            print("PDP için özellik önemi verisi (SHAP) bulunamadı.")


    except Exception as e_detail:
        print(f"Detaylı analiz (SHAP/PDP) sırasında hata: {type(e_detail).__name__}: {e_detail}")
else:
    print("\nDetaylı analiz (SHAP/PDP) için en iyi model belirlenemedi veya uygun değil.")


# ÖRNEK TAHMİN (Mevcut kodunuzdaki gibi)
if best_model_overall_pipeline:
    print(f"\n📦 {best_model_overall_name} Modeli ile Örnek Tahmin:")
    # ... (Mevcut örnek tahmin kodunuz buraya, X_test'in ham halini kullanarak) ...

print("\n--- ANALİZ VE MODELLEME TAMAMLANDI ---")
