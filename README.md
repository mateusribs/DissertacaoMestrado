# Dissertação Mestrado
 Repositório para controle de versão dos códigos referentes ao projeto de mestrado. Os códigos contemplam a detecção e rastreamento de objetos, estimação de estados e aplicação do filtro de Kalman.

# Etapas Alcançadas:
 - (1) Algoritmo baseado em Segmentação para detecção de marcas artificiais na imagem (círculos coloridos);
 - (2) Algoritmo de calibração da câmera para obter os parâmetros intrínsecos da mesma;
 - (3) Algoritmo de estimação da posição e atitude do objeto em questão a partir dos pontos encontrados pelo algoritmo de detecção de marcas artificiais (parâmetros extrínsecos) da câmera;
 - (4) Algoritmo para encontrar a matriz de transformação entre o referencial do objeto e o referencial inercial (solo);
 - Aplicação dos algoritmos propostos em um ambiente virtual desenvolvido no Panda3D para Python. O código pode ser encontrado e simulado no repositório https://github.com/mateusribs/UAV_3d_virtual_env;
 - Aplicação dos algoritmos (1) e (2) em tempo real utilizando uma webcam de baixo custo;
 - Aplicação do algoritmo (3) em tempo real utilizando uma webcam de baixo custo:
 
