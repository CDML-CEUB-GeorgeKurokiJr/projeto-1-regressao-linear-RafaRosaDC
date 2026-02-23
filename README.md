# Previsão de Aluguel de Bicicletas (Bike Sharing) 
Este repositório contém um pipeline completo de Machine Learning e Deep Learning utilizando PyTorch para prever a demanda horária de aluguel de bicicletas. O projeto abrange desde a Análise Exploratória de Dados (EDA) até o desenvolvimento de uma Rede Neural Artificial (MLP) regularizada, com técnicas de Feature Engineering e Early Stopping.

## Tecnologias e Bibliotecas Utilizadas
**Linguagem:** Python

**Deep Learning:** PyTorch (torch, torch.nn, torch.optim)

**Manipulação de Dados:** Pandas, NumPy

**Machine Learning:** Scikit-Learn (StandardScaler, train_test_split, r2_score)

**Visualização:** Matplotlib, Seaborn

## Dicionário de Dados
O modelo utiliza o dataset hour.csv (baseado no clássico Bike Sharing Dataset da UCI). Abaixo está a descrição de todas as variáveis originais e as criadas durante o processo de Feature Engineering:
| Variável | Descrição |
| :--- | :--- |
| `instant` | Identificador único do registro (removido no treinamento). |
| `dteday` | Data do registro no formato string (removida no treinamento). |
| `season` | Estação do ano (1: Primavera, 2: Verão, 3: Outono, 4: Inverno). |
| `yr` | Ano (0: 2011, 1: 2012). |
| `mnth` | Mês do ano (1 a 12). |
| `hr` | Hora do dia (0 a 23). |
| `holiday` | Indica se o dia é feriado (1: Sim, 0: Não). |
| `weekday` | Dia da semana (0: Domingo a 6: Sábado). |
| `workingday` | Indica se é dia útil (1: Sim, 0: Final de semana/Feriado). |
| `weathersit` | Condição climática (1: Limpo/Nublado, 2: Névoa, 3: Neve/Chuva leve, 4: Tempestade). |
| `temp` | Temperatura normalizada (em Celsius). |
| `atemp` | Sensação térmica normalizada. |
| `hum` | Umidade normalizada. |
| `windspeed` | Velocidade do vento normalizada. |
| `casual` | Número de usuários casuais (causa vazamento de dados, removida). |
| `registered` | Número de usuários registrados (causa vazamento de dados, removida). |
| **`cnt`** | **Variável Alvo (Target) - Total de bicicletas alugadas na referida hora.** |

Variáveis Criadas (Feature Engineering)
| Variável | Descrição| 
| :--- | :--- |
|`hr_sin / hr_cos` | Representação trigonométrica (seno e cosseno) da variável hora para capturar a ciclicidade do tempo.| 
| `temp_hr` | Variável de interação: `temperatura` X `hr_sin`. |
| `humidity_temp` | Variável de interação: `umidade` X `temperatura`.|

## Arquitetura e Etapas do Código
O script principal está dividido em 9 etapas lógicas definidas.  Apresento aqui o resumo do que ocorre em cada uma delas:

### **1. Carregamento e Análise Exploratória (EDA)**
* O dataset é importado utilizando o `pandas`.

* É gerado um relatório automático exibindo as dimensões, tipos de dados, valores nulos, estatísticas descritivas e correlação linear com a variável alvo (`cnt`).

* São gerados três gráficos fundamentais: a distribuição da variável alvo, a curva média de aluguéis por hora (evidenciando os horários de pico) e um Heatmap de correlação geral.

### **2. Feature Engineering (Engenharia de Atributos)**
Esta é uma das partes mais sofisticadas do código:

* Codificação Cíclica: Como as horas do dia formam um ciclo contínuo (23h está perto da 0h), a variável `hr` é convertida em coordenadas de seno e cosseno.

* Interações: Criam-se novas variáveis multiplicando clima e horário, ajudando a rede a entender contextos (ex: frio de madrugada vs frio ao meio-dia).

* Prevenção de Data Leakage: As colunas `casual` e `registered` são excluídas para que o modelo não "trapaceie" (pois sua soma resulta na variável alvo cnt).

* One-Hot Encoding: Variáveis categóricas recebem o tratamento pd.get_dummies para evitar que o modelo crie falsas grandezas matemáticas (ex: achar que Inverno(4) vale 4x mais que Primavera(1)).

### **3 e 4. Divisão e Escalonamento**

* Os dados são divididos em 80% para treino e 20% para teste.

* Utiliza-se o `StandardScaler` para normalizar tanto as features (X) quanto o target (y). Isso acelera a convergência do Gradiente Descendente e melhora a precisão da rede neural.

### **5. Conversão para Tensores**

* Transformação dos arrays do NumPy para Tensors do PyTorch (`torch.float32`), habilitando o processamento matricial eficiente necessário para o treinamento da rede.

### **6. Arquitetura do Modelo (MLP Regularizado)**

A rede neural `MLPRegressor` possui camadas ocultas projetadas para capturar relações não lineares complexas:

* Estrutura: Camada Linear (128) -> ReLU -> Linear (64) -> ReLU -> Linear (1).

* Regularização Pesada: Para evitar overfitting (decoração dos dados de treino), a rede utiliza `BatchNorm1d` (normalização em lote) e `Dropout` (desativa aleatoriamente 30% e 20% dos neurônios a cada iteração).

* Função de Custo e Otimizador: Foi escolhida a `HuberLoss`, que é mais robusta a outliers do que o tradicional MSE. O otimizador utilizado é o `Adam` com `weight_decay` (Regularização L2).

### **7. Treinamento com Early Stopping**

* O modelo é treinado por até 1000 épocas.

* Em cada época, avaliamos o erro nos dados de validação/teste.

* Early Stopping: Se o erro de validação parar de cair por 50 épocas consecutivas (`patience=50`), o treinamento é interrompido automaticamente. Os pesos da melhor época são salvos em `best_model.pth`.

### **8. Avaliação Final**

As predições e os valores reais passam por `inverse_transform` (retornando à escala real de "quantidade de bicicletas") e as seguintes métricas são extraídas:

* MAE (Mean Absolute Error): Erro médio em unidades reais (bicicletas).

* MSE (Mean Squared Error): Erro quadrático médio.

* R² (R-Squared): Coeficiente de determinação (percentual de variância explicada pelo modelo).

### **9. Visualizações Finais**

O script exibe um painel com 3 gráficos para atestar a qualidade do treinamento:

* Curva de Loss: Comparação entre o Erro de Treino e o de Validação para avaliar se houve overfitting.

* Real vs Predito: Um gráfico de dispersão com uma linha de predição perfeita ideal ($y = x$).

* Distribuição dos Resíduos: Histograma que avalia a normalidade dos erros cometidos pelo modelo.

## Como Executar

1. Clone este repositório.

2. Certifique-se de ter o arquivo `hour.csv` no mesmo diretório do script principal.

3. Instale as dependências executando:

`pip install pandas numpy torch matplotlib seaborn scikit-learn`

4. Execute o script Python principal:

`python Bike_Sharing_MLP_Report.py`

