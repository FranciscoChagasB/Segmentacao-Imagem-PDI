import cv2
import numpy as np
import matplotlib.pyplot as plt

# Caminhos das imagens
homem_imagem = "homem.jpg"
mulher_imagem = "mulher.jpg"

# Função para exibir os resultados lado a lado
def exibir_resultados(original, resultados, titulos):
    plt.figure(figsize=(20, 10))
    plt.subplot(1, len(resultados) + 1, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Imagem Original")
    plt.axis("off")
    for i, (resultado, titulo) in enumerate(zip(resultados, titulos)):
        plt.subplot(1, len(resultados) + 1, i + 2)
        if len(resultado.shape) == 2:  # Grayscale
            plt.imshow(resultado, cmap="gray")
        else:
            plt.imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
        plt.title(titulo)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# Carregar a imagem
def carregar_imagem(caminho):
    return cv2.imread(caminho)

# 1. Segmentação por Cor (HSV)
def segmentacao_por_cor(imagem):
    # Converter para HSV
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    
    # Ajustar limites para pele e cabelo
    lower_skin = np.array([0, 20, 50], dtype=np.uint8)  # Inclui tons mais escuros
    upper_skin = np.array([35, 255, 255], dtype=np.uint8)  # Amplia a faixa
    
    # Criar máscara inicial
    mascara = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Refinar a máscara com operações morfológicas
    kernel = np.ones((1, 1), np.uint8)
    mascara_refinada = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    
    # Expandir a máscara para incluir o cabelo
    mascara_expandida = cv2.dilate(mascara_refinada, kernel, iterations=1)
    
    # Aplicar a máscara à imagem original
    segmentada = cv2.bitwise_and(imagem, imagem, mask=mascara_expandida)
    
    return mascara_expandida, segmentada

# 2. Detecção de Bordas e Preenchimento Interno
def segmentacao_por_bordas_preenchimento(imagem):
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # Ajuste dos parâmetros do Canny
    bordas = cv2.Canny(gray, 100, 375)  # Limites ajustados para menos bordas
    
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

def segmentacao_por_threshold(imagem):
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    segmented_adaptive = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return segmented_adaptive

# Aplicar as técnicas
def aplicar_tecnicas(caminho_imagem):
    imagem = carregar_imagem(caminho_imagem)
    # Técnica 1: Segmentação por Cor
    mascara_cor, segmentada_cor = segmentacao_por_cor(imagem)
    # Técnica 2: Segmentação por Bordas com Preenchimento
    preenchido = segmentacao_por_bordas_preenchimento(imagem)
    # Técnica 3: K-Means Clustering
    kmeans_segmentacao = segmentacao_por_kmeans(imagem)
    # Técnica 4: Segmentação por Limiar (Otsu)
    limiar = segmentacao_por_otsu(imagem)
    # Técnica 5: Threshold Adaptativo
    threshold = segmentacao_por_threshold(imagem)
    
    # Exibir resultados
    resultados = [mascara_cor, preenchido, kmeans_segmentacao, limiar, threshold]
    titulos = [
        "Máscara (Cor)",
        "Bordas",
        "K-Means Clustering",
        "Segmentação (Otsu)",
        "Segmentação (Threshold Adaptativo)"
    ]
    exibir_resultados(imagem, resultados, titulos)

# Processar imagens
aplicar_tecnicas(homem_imagem)