# Inspeção Automatizada de Defeitos Têxteis
*Pipeline Híbrido de Visão Computacional e Deep Learning (U-Net + CNN)*

Este repositório contém a implementação completa do sistema de detecção e classificação de defeitos têxteis, desenvolvido para operar em ambientes industriais (simulados).

## Como Rodar o Simulador Industrial
O projeto já inclui os modelos treinados (pesos `.pth`). Você pode rodar a simulação industrial sem precisar baixar o dataset completo.

```bash
python simulacao_rolo.py
```

## Como Treinar do Zero (Dataset AITEX)
Para evitar problemas de tamanho de arquivo durante a entrega, o **AITEX Fabric Image Database** não está incluso neste arquivo. Se o avaliador desejar rodar o pipeline de treinamento completo a partir das imagens brutas, siga os passos abaixo:

1. Faça o download do dataset no link oficial: [AITEX Fabric Image Database](https://www.aitex.es/afid/)
2. Extraia o conteúdo e coloque a pasta na raiz deste projeto com o nome EXATO de: `aitex-fabric-image-database`
3. Certifique-se de que a estrutura ficou correta: `aitex-fabric-image-database/Defect_images/` etc.

Após colocar o dataset na pasta correta, os seguintes comandos estarão disponíveis:

```bash
python main.py --dataset

python train_classifier.py

python train_unet_gpu_v3.py
```

## Estrutura Principal do Projeto
* `main.py`: Ponto de entrada para extrair as métricas e relatórios processando o dataset inteiro em lote.
* `simulacao_rolo.py`: Interface GUI que emula uma câmera industrial monitorando tecido em tempo real.
* `resultados/models/`: Contém os pesos vitais previamente treinados (`unet_best.pth`, `defect_classifier.pth`). Sem eles a inferência exige retreinamento.
* `resultados/`: Contém os gráficos de perda, a matriz de confusão do classificador e os resultados qualitativos.
