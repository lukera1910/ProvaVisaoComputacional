# ProvaVisaoComputacional

1 - Descrição do problema
Implementar um pipeline que contenha carregamento e processamento de imagens (redimensionamento, filtro gaussiano e equalização de historiograma), extrair as caracteristícas e fazer uma classificação usando um modelo de IA.

2 - Justificativa das técnicas utilizadas
Usei redimensionamento para padronizar o tamanho das imagens, reduzir o custo computacional e evitar distorções na rede causadas por tamanhos diferentes de imagens.
Usei o filtro gaussiano para reduzir o ruído (o que atrapalha o aprendizado da rede neural), suavizar as bordas e melhorar a robustez de classificação.
Sobre o historiograma, ele melhora o contraste da imagem, uniformiza a distribuição do brilho e facilita a detecção de contornos e formas pelas IA.

3 - Etapas realizadas
1° Etapa: Fiz uma função para carregar as minhas imagens
2° Etapa: Criei um dicionário onde armazenei o caminho das pastas
3° Etapa: Carreguei e filtrei o CIFAR-10 (para apenas gatos e cachorros)
4° Etapa: Juntei meus dados + CIFAR-10
5° Etapa: Embaralhei e dividi os dados de treino e teste
6° Etapa: Criei a CNN
7° Etapa: Treinei o modelo
8° Etapa: Avaliação do modelo

4 - Resultados obtidos
Bom, a acurácia geral foi de 60%, ele acerta de 6 em cada 10 imagens, creio eu que seja razoável, para a classe Gato, a precisão tá alta (0.65) e o recall é baixo (0.50), ele deixa de identificar muitos gatos corretamente, e o F1-score é moderado (0.56), há equilibrio entre os dois, para a classe Cachorro, o recall é alto (0.72), mas ele erra demais, precisão (0.57), o F1-score é um pouco melhor (0.63)

5 - Tempo total gasto
Longas 2h40min de duração.

6 - Dificuldades encontradas
Foi no treinamento, como executei no Colab, a RAM é limitada, tive que diminuir a quantidade de imagens que o CIFAR tava usando, e também diminuir o batch_size pra 16 também, porque sempre dava erro por causa da RAM