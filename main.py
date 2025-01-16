import cv2
import numpy as np
import matplotlib.pyplot as plt

# Caminhos das imagens
homem_imagem1 = "./entrada/homem.jpg"
mulher_imagem1 = "./entrada/mulher.jpg"
homem_imagem2 = "./entrada/homem_imagem.jpg"
mulher_imagem2 = "./entrada/mulher_imagem.jpg"
homem_imagem3 = "./entrada/homem_imagem2.jpg"
mulher_imagem3 = "./entrada/mulher_imagem2.jpg"

# Função para exibir os resultados lado a lado
def exibir_resultados(original, resultados, titulos):
    plt.figure(figsize=(15, 10))  # Ajustar o tamanho do plot
    imagens = [original] + resultados
    titulos = ["Imagem Original"] + titulos

    for i, (imagem, titulo) in enumerate(zip(imagens, titulos)):
        plt.subplot(3, 2, i + 1)  # Layout 3x2
        if len(imagem.shape) == 2:  # Grayscale
            plt.imshow(imagem, cmap="gray")
        else:
            plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
        plt.title(titulo)
        plt.axis("off")
    
    plt.tight_layout()
    plt.show(block=False)  # Abre todas as janelas sem bloquear a execução

# Carregar a imagem
def carregar_imagem(caminho):
    return cv2.imread(caminho)

# 1. Segmentação por Cor (HSV)
def segmentacao_por_cor(imagem):
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 50], dtype=np.uint8)
    upper_skin = np.array([35, 255, 255], dtype=np.uint8)
    mascara = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = np.ones((1, 1), np.uint8)
    mascara_refinada = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    mascara_expandida = cv2.dilate(mascara_refinada, kernel, iterations=1)
    segmentada = cv2.bitwise_and(imagem, imagem, mask=mascara_expandida)
    return mascara_expandida, segmentada

# 2. Detecção de Bordas e Preenchimento Interno
def segmentacao_por_bordas_preenchimento(imagem):
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    bordas = cv2.Canny(gray, 100, 375)
    return bordas

# 3. K-Means Clustering
def segmentacao_por_kmeans(imagem, k=2):
    Z = imagem.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    resultado = centers[labels.flatten()]
    return resultado.reshape(imagem.shape)

# 4. Segmentação Baseada em Limiar (Otsu)
def segmentacao_por_otsu(imagem):
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    _, limiar = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return limiar

# 5. Threshold Adaptativo
def segmentacao_por_threshold(imagem):
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    segmented_adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return segmented_adaptive

# Aplicar as técnicas
def aplicar_tecnicas(caminho_imagem):
    imagem = carregar_imagem(caminho_imagem)
    mascara_cor, segmentada_cor = segmentacao_por_cor(imagem)
    preenchido = segmentacao_por_bordas_preenchimento(imagem)
    kmeans_segmentacao = segmentacao_por_kmeans(imagem)
    limiar = segmentacao_por_otsu(imagem)
    threshold = segmentacao_por_threshold(imagem)
    resultados = [mascara_cor, preenchido, kmeans_segmentacao, limiar, threshold]
    titulos = [
        "Máscara (Cor)",
        "Bordas",
        "K-Means Clustering",
        "Segmentação (Otsu)",
        "Segmentação (Threshold Adaptativo)",
    ]
    exibir_resultados(imagem, resultados, titulos)

# Processar imagens
imagens = [
    homem_imagem1,
    homem_imagem2,
    homem_imagem3,
    mulher_imagem1,
    mulher_imagem2,
    mulher_imagem3,
]

for img in imagens:
    aplicar_tecnicas(img)

plt.show()