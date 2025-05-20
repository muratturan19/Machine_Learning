import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2 # 1️⃣ Transfer Learning için
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # 3️⃣ EarlyStopping & LR Ayarı için
import matplotlib.pyplot as plt
import numpy as np
import os
import shap # 4️⃣ SHAP için (eğer yüklü değilse !pip install shap ile yükleyin)

# 1. Google Drive'ı Colab'a Bağlama (Zaten yapılmış ve çalışır durumda varsayıyorum)
# from google.colab import drive
# drive.mount('/content/drive')

# 2. Veri Seti Yolu ve Parametreler
data_dir = '/content/drive/MyDrive/bottle' # Sizin klasör yolunuz
batch_size = 16 # Veri seti küçük olduğu için batch size'ı biraz düşürebiliriz
img_height = 224 # MobileNetV2 için genellikle 224x224
img_width = 224
validation_split_ratio = 0.2 # Veri setinin %20'si doğrulama için

# Epoch sayıları (EarlyStopping ile yönetilecek)
num_epochs_initial = 75 # Başlangıç ve transfer öğrenme için maksimum epoch
# num_epochs_finetune = 30 # İnce ayar için maksimum epoch (şimdilik kullanmıyoruz)

# 3. Veri Setini Yükleme ve Hazırlama
print("Eğitim verileri yükleniyor...")
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=validation_split_ratio,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  label_mode='int'
)

print("Doğrulama verileri yükleniyor...")
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=validation_split_ratio,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  label_mode='int'
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Bulunan dosyalar: {sum(1 for _ in train_ds.unbatch()) + sum(1 for _ in val_ds.unbatch())}, Sınıflar: {class_names}, Sınıf sayısı: {num_classes}")

# Veri setlerini optimize etme
AUTOTUNE = tf.data.AUTOTUNE
def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=500) # Buffer size'ı veri sayısına göre ayarlayabiliriz
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds) # Doğrulamayı shuffle etmeye gerek yok, ama cache ve prefetch faydalı

# 4. Veri Ön İşleme ve 2️⃣ Data Augmentation
# MobileNetV2 piksellerin [-1, 1] aralığında olmasını bekler.
# Data augmentation'ı modelin bir parçası olarak ekleyelim.

data_augmentation_layers = keras.Sequential(
  [
    layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2), # Kontrast artırmasını ekledik
    # layers.RandomBrightness(0.2) # İsteğe bağlı parlaklık
  ],
  name="data_augmentation"
)

# 5. 1️⃣ Transfer Learning ile CNN Modelini Oluşturma

# Temel modeli yükle (MobileNetV2)
base_model = MobileNetV2(input_shape=(img_height, img_width, 3),
                         include_top=False,  # Son sınıflandırıcı katmanı olmadan
                         weights='imagenet') # ImageNet ağırlıklarıyla

# Temel modelin katmanlarını dondur (başlangıçta eğitilmesinler)
base_model.trainable = False

# Kendi sınıflandırıcı katmanlarımızı oluşturalım
# Giriş katmanını preprocess_input ve data_augmentation ile hazırlayalım
inputs = keras.Input(shape=(img_height, img_width, 3))
x = data_augmentation_layers(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x) # MobileNetV2'nin kendi ön işleme katmanı
x = base_model(x, training=False) # Temel modeli dondurulmuş ağırlıklarla çalıştır
x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
x = layers.Dropout(0.3, name="top_dropout_1")(x) # Önceki 0.2'den biraz artırdık
x = layers.Dense(128, activation='relu', name="dense_1")(x)
# x = layers.BatchNormalization()(x) # İsteğe bağlı: Batch Norm eklenebilir
x = layers.Dropout(0.5, name="top_dropout_2")(x) # Önceki 0.5 iyiydi
outputs = layers.Dense(num_classes, activation='softmax', name="output_layer")(x)

model = keras.Model(inputs, outputs, name="transfer_learning_model_bottles")

# 6. Modeli Derleme
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Başlangıç için Adam
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()

# 7. 3️⃣ EarlyStopping & LR Ayarı (Callback'ler)
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=10,        # val_loss 10 epoch iyileşmezse dur
                               restore_best_weights=True, # En iyi ağırlıkları geri yükle
                               verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,         # Öğrenme oranını %20'sine düşür
                              patience=5,         # val_loss 5 epoch iyileşmezse
                              min_lr=0.00001,     # Minimum öğrenme oranı
                              verbose=1)

# 8. Modeli Eğitme (Sadece üst katmanlar - base_model.trainable=False)
print("\n--- Transfer Öğrenme Aşaması (Üst Katmanların Eğitimi) ---")
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=num_epochs_initial, # Maksimum epoch, EarlyStopping yönetecek
  callbacks=[early_stopping, reduce_lr]
)

# Eğitilmiş modelin performansını değerlendirelim
print("\nTransfer öğrenme sonrası doğrulama seti üzerinde değerlendirme:")
loss, accuracy = model.evaluate(val_ds)
print(f"Doğrulama Kaybı (Son): {loss:.4f}")
print(f"Doğrulama Başarımı (Son): {accuracy:.4f}")

# En iyi epoch'taki performansı da yazdıralım (restore_best_weights=True sayesinde)
if early_stopping.best_epoch > 0 : # Eğer erken durduysa
    print(f"En iyi epoch (val_loss): {early_stopping.best_epoch + 1}") # Epoch'lar 0'dan başlar
    print(f"En iyi val_loss: {early_stopping.best:.4f}")
    # val_accuracy için ayrıca loglamadıysak direkt alamayız ama val_loss'a karşılık gelen iyidir.
else: # Erken durmadıysa, son epoch en iyisidir.
    print("Eğitim erken durmadı, son epoch değerleri en iyi kabul ediliyor.")

# 9. Eğitim Sonuçlarını Görselleştirme
def plot_history(history_data, title_prefix=""):
    acc = history_data.history['accuracy']
    val_acc = history_data.history['val_accuracy']
    loss_hist = history_data.history['loss']
    val_loss_hist = history_data.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Eğitim Başarımı')
    plt.plot(epochs_range, val_acc, label='Doğrulama Başarımı')
    plt.axvline(early_stopping.best_epoch if early_stopping.best_epoch > 0 else len(acc) -1 , color='r', linestyle='--', label=f'En İyi Epoch (val_loss)')
    plt.legend(loc='lower right')
    plt.title(f'{title_prefix}Eğitim ve Doğrulama Başarımı')
    plt.xlabel('Epoch')
    plt.ylabel('Başarım (Accuracy)')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss_hist, label='Eğitim Kaybı')
    plt.plot(epochs_range, val_loss_hist, label='Doğrulama Kaybı')
    plt.axvline(early_stopping.best_epoch if early_stopping.best_epoch > 0 else len(loss_hist) -1, color='r', linestyle='--', label=f'En İyi Epoch (val_loss)')
    plt.legend(loc='upper right')
    plt.title(f'{title_prefix}Eğitim ve Doğrulama Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp (Loss)')
    plt.tight_layout()
    plt.show()

plot_history(history, "Transfer Öğrenme ")


# 10. 4️⃣ Görsel Açıklama (SHAP)
# NOT: SHAP için !pip install shap komutunu Colab'da çalıştırmanız gerekebilir.
print("\n--- SHAP ile Model Yorumlama ---")

# SHAP, modelin eğitim verilerinden bazı örnekler üzerinde nasıl çalıştığını görmek için kullanılır.
# Genellikle tüm eğitim setini kullanmak yerine bir alt küme (background dataset) kullanılır.
# SHAP'ın GradientExplainer'ı TensorFlow 2.x modelleriyle iyi çalışır.

# Arka plan veri seti için eğitim setinden birkaç batch alalım
background_images_list = []
background_labels_list = []
# Bellek sorunlarını önlemek için az sayıda örnek alalım
# train_ds'i unbatch edip take ile almak daha kontrollü olabilir.
# Veya doğrudan train_ds'den birkaç batch alalım
for images, labels in train_ds.take(5): # Örn: 5 batch (5 * batch_size kadar örnek)
    background_images_list.append(images.numpy())
    background_labels_list.append(labels.numpy())

if not background_images_list:
    print("SHAP için arka plan verisi alınamadı. train_ds'i kontrol edin.")
else:
    background_images = np.concatenate(background_images_list, axis=0)
    # background_labels = np.concatenate(background_labels_list, axis=0) # Etiketlere şimdilik gerek yok
    print(f"SHAP için arka plan veri seti boyutu: {background_images.shape}")


    # Açıklamak istediğimiz görselleri doğrulama setinden alalım
    test_images_list = []
    test_labels_list = []
    for images, labels in val_ds.take(1): # Bir batch alalım (örneğin batch_size kadar)
        test_images_list.append(images.numpy())
        test_labels_list.append(labels.numpy())

    if not test_images_list:
        print("SHAP için test verisi alınamadı. val_ds'i kontrol edin.")
    else:
        test_images_to_explain = np.concatenate(test_images_list, axis=0)
        test_labels_to_explain = np.concatenate(test_labels_list, axis=0)
        print(f"Açıklanacak test görselleri sayısı: {test_images_to_explain.shape[0]}")

        # SHAP GradientExplainer oluştur
        # ÖNEMLİ: SHAP, modelin softmax öncesi çıktılarını (logitleri) bekleyebilir.
        # Eğer modeliniz softmax ile bitiyorsa, ya softmax'sız bir model oluşturun
        # ya da explainer'ın bu durumu nasıl ele aldığına dikkat edin.
        # GradientExplainer genellikle modelin kendisini alır.
        # Eğer modelin içinde data_augmentation ve preprocess_input varsa, explainer'a
        # ham görselleri (0-255 aralığında) vermemiz gerekir.
        # Modelimiz zaten bu ön işlemeyi yaptığı için sorun olmamalı.

        explainer = shap.GradientExplainer(model, background_images)

        # Seçilen test görselleri için SHAP değerlerini hesapla
        # Az sayıda görsel seçelim (örneğin ilk 3 tane), çünkü hesaplama zaman alabilir
        num_explanations = min(3, test_images_to_explain.shape[0])
        if num_explanations > 0:
            shap_values = explainer.shap_values(test_images_to_explain[:num_explanations])
            print(f"SHAP değerleri hesaplandı. Boyut: {len(shap_values)} (sınıf sayısı kadar) x {shap_values[0].shape}")

            # SHAP değerlerini görselleştir
            # shap.image_plot, ham görselleri ve SHAP değerlerini bekler.
            # Görsellerimiz şu an [-1, 1] aralığında olabilir (MobileNetV2 preprocess_input sonrası)
            # Veya [0,1] (rescaling sonrası). SHAP'ın beklentisine göre ayarlamak gerekebilir.
            # `shap.image_plot` piksellerin [0,1] veya [0,255] olmasını bekleyebilir.
            # `test_images_to_explain` zaten [0,255] aralığında yüklenmişti, model içindeki
            # katmanlar bunları işliyor. Bu yüzden doğrudan kullanabiliriz.
            # Ancak, preprocess_input [-1,1]'e çevirdiği için SHAP görselleştirmesi için
            # [0,1]'e geri getirmek daha doğru olabilir.
            # Modelimiz Data Augmentation ve Rescaling (preprocess_input) içerdiği için,
            # SHAP explainer'a ham görselleri ([0, 255] aralığında) vermeliyiz.
            # `test_images_to_explain` zaten bu formatta olmalı (image_dataset_from_directory'den geldiği için)

            # Görselleştirme için görselleri [0,1] aralığına getirelim
            display_test_images = test_images_to_explain[:num_explanations] / 255.0

            print("\nSHAP Görselleştirmeleri:")
            shap.image_plot(shap_values, display_test_images, labels=[f"Class {i} ({class_names[i]})" for i in range(num_classes)])
            plt.suptitle("SHAP Değerleri ile Model Yorumlama (İlk Birkaç Test Görseli)", y=0.92) # Ana başlık
            plt.show()

            # Her sınıf için tahminler (logitler veya olasılıklar)
            # model.predict ile tahminleri alıp, hangi sınıfın SHAP değerlerine baktığımızı netleştirebiliriz.
            predictions = model.predict(test_images_to_explain[:num_explanations])
            for i in range(num_explanations):
                print(f"\nTest görseli {i+1}:")
                plt.imshow(test_images_to_explain[i].astype("uint8"))
                plt.title(f"Gerçek Sınıf: {class_names[test_labels_to_explain[i]]}")
                plt.axis('off')
                plt.show()
                for j, class_name in enumerate(class_names):
                    print(f"  Tahmin ({class_name}): {predictions[i][j]:.4f}")
        else:
            print(f"SHAP için açıklanacak yeterli test görseli bulunamadı ({num_explanations}).")

# 11. (İSTEĞE BAĞLI) Fine-Tuning (İnce Ayar)
# Önceki sonuçlara göre bu adım değerlendirilebilir.
# print("\n--- İnce Ayar Aşaması (Opsiyonel) ---")
# base_model.trainable = True # Tüm temel modeli veya bazı katmanlarını çöz
# fine_tune_at = 100 # Örnek: MobileNetV2'nin son ~54 katmanını eğitilebilir yap
# for layer in base_model.layers[:fine_tune_at]:
#    layer.trainable = False
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # ÇOK DÜŞÜK öğrenme oranı
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#               metrics=['accuracy'])
# model.summary()
# history_fine_tune = model.fit(train_ds, epochs=num_epochs_initial + num_epochs_finetune,
#                               initial_epoch=history.epoch[-1] +1,
#                               validation_data=val_ds, callbacks=[early_stopping, reduce_lr])
# loss_ft, accuracy_ft = model.evaluate(val_ds)
# print(f"İnce Ayar Doğrulama Kaybı: {loss_ft:.4f}, Başarımı: {accuracy_ft:.4f}")
# plot_history(history_fine_tune, "İnce Ayar ")

# 12. Model Kaydetme (İsteğe bağlı)
# model_save_path = '/content/drive/MyDrive/bottles_mobilenetv2_shap_model.keras'
# model.save(model_save_path)
# print(f"Model {model_save_path} adresine kaydedildi.")

