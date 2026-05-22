import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage.color import rgb2lab, lab2rgb

# ==========================================
# 1. КОНФИГУРАЦИЯ
# ==========================================

# Путь к Keras модели
KERAS_MODEL_PATH = r'model/colorizer_best.keras'

# Пути к файлам Caffe модели
CAFFE_PROTO = "model/colorization_deploy_v2.prototxt"
CAFFE_MODEL = "model/colorization_release_v2.caffemodel"
CAFFE_PTS = "model/pts_in_hull.npy"

# Настройки страницы
st.set_page_config(page_title='AI Colorizer', layout='wide')
st.title('AI Colorizer')
st.write('Загрузите изображение. Результат будет приведен к читаемому размеру с сохранением пропорций.')

# Максимальный размер стороны результата (чтобы не сохранять 4K разрешение, если не нужно)
MAX_DISPLAY_SIZE = 1024

# Фиксированный баланс смешивания
MIX_RATIO = 0.5


# ==========================================
# 2. ТИХАЯ ЗАГРУЗКА МОДЕЛЕЙ
# ==========================================

@st.cache_resource
def loading():
    """Загружает модели без лишних сообщений"""
    models = {}

    # Keras
    if os.path.exists(KERAS_MODEL_PATH):
        try:
            models['keras'] = load_model(KERAS_MODEL_PATH)
        except:
            pass  # Молчаливый пропуск при ошибке

    # Caffe
    if os.path.exists(CAFFE_PROTO) and os.path.exists(CAFFE_MODEL):
        try:
            net = cv2.dnn.readNetFromCaffe(CAFFE_PROTO, CAFFE_MODEL)
            pts = np.load(CAFFE_PTS)
            pts = pts.transpose().reshape(2, 313, 1, 1)
            layer1 = net.getLayerId("class8_ab")
            layer2 = net.getLayerId("conv8_313_rh")
            net.getLayer(layer1).blobs = [pts.astype("float32")]
            net.getLayer(layer2).blobs = [np.full([1, 313], 2.606, dtype="float32")]
            models['caffe'] = net
        except:
            pass

    return models


models_dict = loading()

if not models_dict:
    st.error("Модели не найдены. Проверьте пути к файлам.")
    st.stop()


# ==========================================
# 3. ФУНКЦИИ ОБРАБОТКИ
# ==========================================

def get_keras_ab_small(img_pil):
    """Возвращает AB каналы (256x256) от Keras"""
    IMG_SIZE = 256
    img_resized = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img_arr = np.array(img_resized, dtype=float) / 255.0
    lab = rgb2lab(img_arr)
    L_channel = lab[:, :, 0]
    X_input = np.stack((L_channel, L_channel, L_channel), axis=-1)
    X_input = (X_input / 50) - 1
    X_input = X_input.reshape(1, IMG_SIZE, IMG_SIZE, 3)
    SATURATION_MULTIPLIER = 1.4
    pred_ab = models_dict['keras'].predict(X_input, verbose=0)[0] * 128 * SATURATION_MULTIPLIER
    return pred_ab


def get_caffe_ab_small(img_pil):
    """Возвращает AB каналы (256x256) от Caffe"""
    open_cv_image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    normalized = open_cv_image.astype("float32") / 255.0
    lab_image = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab_image, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    models_dict['caffe'].setInput(cv2.dnn.blobFromImage(L))
    ab = models_dict['caffe'].forward()[0, :, :, :].transpose((1, 2, 0))
    ab_256 = cv2.resize(ab, (256, 256))
    return ab_256


# ==========================================
# 4. ИНТЕРФЕЙС И ЛОГИКА ПРОПОРЦИЙ
# ==========================================

uploaded_file = st.file_uploader(
    'Загрузите изображение',
    type=['jpeg', 'png', 'jpg', 'bmp']
)

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert('RGB')
    orig_w, orig_h = original_image.size

    # --- РАСЧЕТ ЦЕЛЕВОГО РАЗМЕРА С СОХРАНЕНИЕМ ПРОПОРЦИЙ ---
    if orig_w > orig_h:
        # Если ширина больше - ограничиваем ширину
        new_w = MAX_DISPLAY_SIZE
        new_h = int(orig_h * (MAX_DISPLAY_SIZE / orig_w))
    else:
        # Если высота больше - ограничиваем высоту
        new_h = MAX_DISPLAY_SIZE
        new_w = int(orig_w * (MAX_DISPLAY_SIZE / orig_h))

    target_size = (new_w, new_h)

    if st.button("🎨Раскрасить🎨"):
        with st.spinner('Обработка...'):

            # 1. Получаем цвета в маленьком размере (256x256)
            ab_k_small = get_keras_ab_small(original_image) if 'keras' in models_dict else None
            ab_c_small = get_caffe_ab_small(original_image) if 'caffe' in models_dict else None

            # 2. Смешиваем
            if 'keras' in models_dict and 'caffe' in models_dict:
                ab_hybrid_small = (ab_k_small * MIX_RATIO) + (ab_c_small * (1 - MIX_RATIO))
            elif 'keras' in models_dict:
                ab_hybrid_small = ab_k_small
            else:
                ab_hybrid_small = ab_c_small

            # 3. Растягиваем карту цветов до ЦЕЛЕВОГО размера (с сохранением пропорций)
            ab_target = cv2.resize(ab_hybrid_small, target_size)

            # 4. Получаем яркость (L) в ЦЕЛЕВОМ размере (ресайз оригинала)
            img_resized_target = original_image.resize(target_size, Image.LANCZOS)  # LANCZOS для качества
            lab_target = rgb2lab(np.array(img_resized_target) / 255.0)
            L_target = lab_target[:, :, 0]

            # 5. Собираем финальное изображение
            final_lab = np.zeros((new_h, new_w, 3))
            final_lab[:, :, 0] = L_target
            final_lab[:, :, 1:] = ab_target

            # 6. Конвертация
            rgb_img = lab2rgb(final_lab)
            final_image_pil = Image.fromarray((rgb_img * 255).astype(np.uint8))

            # --- ВЫВОД ---
            col1, col2 = st.columns(2)

            with col1:
                st.image(original_image.resize(target_size), caption="Оригинал (масштабирован)",
                         use_container_width=True)

            with col2:
                st.image(final_image_pil, caption=f"Результат ({new_w}x{new_h})", use_container_width=True)