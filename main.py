import streamlit as st
import numpy as np
from PIL import Image
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

st.title("Image Compression using SVD")
st.write('Приложение, которое позволяет пользователю загрузить изображение и выбрать количество сингулярных чисел для сжатия')

# Загрузка изображения
uploaded_file = st.sidebar.file_uploader("Выберите черно-белое изображение...", type=["jpg", "png"])
if uploaded_file is not None:
    # Конвертация в черно-белое
    image = Image.open(uploaded_file).convert('L')  
    image_array = np.array(image)
    
    st.write("Размер матрицы изображения:", image_array.shape)

    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Разложение по SVD
    U, S, Vt = np.linalg.svd(image_array, full_matrices=False)

    st.sidebar.write('Визуализация сингулярных значений')
    plt.figure(figsize=(10, 5))
    plt.plot(S, marker='o')
    plt.title('Сингулярные значения')
    plt.xlabel('Индекс')
    plt.ylabel('Сингулярное значение')
    plt.grid()
    st.pyplot(plt)

    # Выбор количества сингулярных чисел
    k = st.slider("Выберите количество сингулярных значений (k)", min_value=1, max_value=min(image_array.shape), value=5)
    
    # Сжатие изображения
    S_k = np.zeros((U.shape[0], Vt.shape[0]))
    np.fill_diagonal(S_k, S[:k])
    compressed_image = U @ S_k @ Vt

    # Отображение сжатого изображения
    st.image(compressed_image, caption='Compressed Image', use_column_width=True)

    # Доля k от всех сингулярных чисел
    total_singular_values = len(S)
    fraction_k = k / total_singular_values
    st.write(f"Доля выбранного k от всех сингулярных чисел: {fraction_k:.2f}")

  


    