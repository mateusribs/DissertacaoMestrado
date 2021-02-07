# Dissertação Mestrado
 Repositório para controle de versão dos códigos referentes ao projeto de mestrado. Os códigos contemplam a detecção e rastreamento de objetos, estimação de estados e aplicação do filtro de Kalman. 
 
 Versões de software:
 - Python: 3.8.5;
 - OpenCV Contrib: 4.2.0;
 - Numpy: 1.19.2;
 - Pytorch: 1.5.1.
 
 Hardwares:
 - Webcam Genérica 720p 30FPS;
 - Arduino Nano;
 - MPU9250;

# Etapas Alcançadas:
 ------------------------------- Visão Computacional -----------------------------------------
 - (1) Algoritmo de calibração da câmera para obter os parâmetros intrínsecos através de CharUco;
 - (2) Algoritmo de estimação da posição e atitude do objeto em questão a partir dos pontos encontrados pelo algoritmo de detecção de marcadores artificiais (ArUco) da câmera (parâmetros extrínsicos);
 - (3) Algoritmo para encontrar a matriz de transformação entre o referencial do objeto e o referencial inercial (solo);
 - Aplicação dos algoritmos propostos em um ambiente virtual desenvolvido no Panda3D para Python. O código pode ser encontrado e simulado no repositório https://github.com/mateusribs/UAV_3d_virtual_env;
 - Aplicação dos algoritmos (1) e (2) em tempo real utilizando uma webcam de baixo custo;
 - Aplicação do algoritmo (3) em tempo real utilizando uma webcam de baixo custo:
 
 ------------------------------ IMU ---------------------------------------------------------
 - Calibragem do sensor MPU9250 de 9 eixos (Acelerômetro, Giroscópio e Magnetômetro) - Algoritmo de calibragem baseado no trabalho de Bryan Taylor. Link para o repositório: https://github.com/bolderflight/MPU9250.
   - Acelerômetro ---> Okay
   - Giroscópio ----> Okay
 - Fusão Sensorial - Utiliza-se o algoritmo de AHRS (Attitude and Heading Reference System) proposto por Mahony. O algoritmo foi implementado através da biblioteca SensorFusion disponbilizada por https://github.com/aster94/SensorFusion. Baseada também nos trabalhos de Paul Stroffregen https://github.com/PaulStoffregen/MahonyAHRS. O algoritmo realiza a fusão sensorial entre os dados obtidos pelo acelerômetro e giroscópio através da abordagem de quatérnios para determinação da atitude.
 
 # Etapas Futuras:
  - Melhorar a estimação de atitude, pois há muito ruído envolvendo principalmente os ângulos de rolagem e arfagem (roll e pitch).;
  - Melhorar a calibragem do magnetômetro e verificar se os resultados obtidos (que são insatisfatórios) são provenientes de má calibragem ou defeito no sensor;
  - Propor algoritmo de fusão sensorial entre as informações obtidas com visão computacional e IMU.
 
