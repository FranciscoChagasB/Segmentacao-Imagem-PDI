# Projeto: Segmentação de Imagens com Técnicas Diversificadas

## Introdução

Este projeto foi desenvolvido com o objetivo de implementar, comparar e avaliar diferentes técnicas de segmentação de imagens, com foco em aplicações práticas e análise visual. As técnicas abordadas incluem segmentação por cor, detecção de bordas, K-Means clustering, segmentação por limiar de Otsu e threshold adaptativo.


## Tecnologias Utilizadas

- Python 3
- OpenCV
- Matplotlib
- NumPy


## Técnicas de Segmentação


### Segmentação por Cor
Utiliza o espaço de cor HSV para isolar regiões baseadas em faixas de tonalidades, com refinamento usando operações morfológicas.


### Detecção de Bordas e Preenchimento Interno
Emprega o algoritmo Canny para detectar bordas e explorar as fronteiras de objetos na imagem.


### K-Means Clustering
Um método de clustering que agrupa pixels com base na similaridade de cores, segmentando a imagem em regiões distintas.


### Segmentação por Limiar de Otsu
Calcula automaticamente o melhor limiar para separar objetos do fundo em imagens em tons de cinza.


### Threshold Adaptativo
Ajusta o limiar localmente para lidar com variações de iluminação na imagem.


## Como Executar o Projeto

1. Certifique-se de ter o Python 3 instalado.
2. Instale as dependências necessárias:
   ```bash
   pip install opencv-python numpy matplotlib
   ```
3. Coloque as imagens que deseja processar no diretório do projeto.
Altere os caminhos das imagens no código principal para as imagens desejadas.
Execute o script principal para visualizar os resultados:
   ```bash
   python main.py
   ```
   
## Resultados Esperados
O script gera visualizações lado a lado das imagens originais e as segmentações resultantes de cada técnica. Os resultados destacam regiões relevantes como pele, cabelo e bordas, dependendo da abordagem.


## Referências
* GONZALEZ, Rafael C.; WOODS, Richard E. Processamento de imagens digitais. 4. ed. São Paulo: Pearson, 2018.
* CANNY, John. A computational approach to edge detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, v. 8, n. 6, p. 679-698, 1986.
* MACQUEEN, James. Some methods for classification and analysis of multivariate observations. In: Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability. Berkeley: University of California Press, 1967. p. 281-297.
* OTSU, Nobuyuki. A threshold selection method from gray-level histograms. IEEE Transactions on Systems, Man, and Cybernetics, v. 9, n. 1, p. 62-66, 1979.
* SONKA, Milan; HLAVAC, Vaclav; BOYLE, Roger. Image processing, analysis, and machine vision. 4. ed. Cengage Learning, 2014.


## Contribuições
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests com melhorias ou novas ideias.

