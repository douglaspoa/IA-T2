# Pacman AI Agents

Este projeto implementa agentes de IA para o jogo Pacman usando técnicas de inferência probabilística e algoritmos de decisão. Os agentes são divididos em ofensivos e defensivos, cada um com estratégias específicas para maximizar a pontuação do time.

## Sumário

- [Classes](#classes)
  - [InferenceAgent](#inferenceagent)
  - [OffensiveAgent](#offensiveagent)
  - [DefensiveAgent](#defensiveagent)
- [Como Executar](#como-executar)
- [Licença](#licença)

## Classes

### InferenceAgent

A classe `InferenceAgent` é responsável por inferir as posições dos agentes adversários no jogo Pacman usando técnicas de inferência probabilística. Ela utiliza um modelo de crenças para manter uma estimativa das posições dos inimigos com base em observações ruidosas e nas possíveis transições de estado.

#### Funcionalidades

1. **`registerInitialState(self, gameState)`**
   - Inicializa o estado do agente.
   - Define a posição inicial, as posições legais no tabuleiro e os índices dos agentes adversários.
   - Inicializa as crenças para cada agente adversário, assumindo que eles estão em suas posições iniciais com probabilidade 1.

2. **`initializeBeliefs(self, enemy)`**
   - Inicializa uma distribuição de crença uniforme para um inimigo específico, assumindo que ele pode estar igualmente em qualquer posição legal.

3. **`elapseTime(self, enemy, gameState)`**
   - Atualiza a crença sobre a posição de um inimigo ao longo do tempo, considerando todas as possíveis transições de estado (movimentos) que o inimigo pode fazer.

4. **`observe(self, enemy, observation, gameState)`**
   - Atualiza a crença sobre a posição de um inimigo com base em uma observação ruidosa da distância ao inimigo.

5. **`chooseAction(self, gameState)`**
   - Escolhe a melhor ação para o agente com base nas crenças atualizadas sobre as posições dos inimigos.
   - Utiliza o algoritmo expectimax para calcular a utilidade esperada das ações e escolhe a ação que maximiza essa utilidade.

6. **`maxFunction(self, gameState, depth)`**
   - Parte do algoritmo expectimax, esta função escolhe a ação que maximiza a utilidade esperada para o agente.

7. **`expectiFunction(self, gameState, enemy, depth)`**
   - Parte do algoritmo expectimax, esta função calcula a utilidade esperada considerando as ações dos inimigos.

8. **`enemyDistances(self, gameState)`**
   - Retorna as distâncias aos inimigos, usando a posição mais provável inferida quando a posição exata não é conhecida.

9. **`evaluationFunction(self, gameState)`**
   - Função de avaliação personalizada que calcula a utilidade de um estado do jogo (não implementada neste trecho de código).

### OffensiveAgent

A classe `OffensiveAgent` é uma especialização da classe `InferenceAgent` e representa um agente ofensivo no jogo Pacman. Seu objetivo principal é coletar alimentos na metade do tabuleiro do oponente e retornar à base sem ser capturado.

#### Funcionalidades

1. **`registerInitialState(self, gameState)`**
   - Chama o método `registerInitialState` da classe `InferenceAgent` para inicializar o estado do agente.
   - Inicializa a variável `retreating` para determinar quando o agente deve retornar à base.

2. **`chooseAction(self, gameState)`**
   - Calcula o tempo restante de medo dos fantasmas adversários (`scaredTimes`).
   - Obtém a pontuação atual do jogo.
   - Define o limite de alimentos a serem coletados antes de retornar à base (`carryLimit`), ajustando-o com base na pontuação do jogo.
   - Decide se o agente deve continuar coletando alimentos ou retornar à base com base no número de alimentos carregados e no tempo de medo dos fantasmas adversários.
   - Chama o método `chooseAction` da classe `InferenceAgent` para selecionar a melhor ação com base nas crenças atualizadas sobre as posições dos inimigos.

3. **`evaluationFunction(self, gameState)`**
   - Avalia a utilidade de um estado do jogo para o agente ofensivo.
   - Considera vários fatores, incluindo:
     - Posição atual do agente (`myPos`).
     - Alimentos disponíveis no tabuleiro (`targetFood`).
     - Distância até a metade do tabuleiro (`distanceFromStart`).
     - Distâncias aos fantasmas inimigos (`ghostDistances`).
     - Distância mínima aos fantasmas (`minGhostDistances`).
     - Distâncias às cápsulas de poder (`capsulesChasingDistances`).
     - Tempo de medo dos fantasmas (`scaredTimes`).
   - Retorna uma pontuação baseada nesses fatores, ajustando o comportamento do agente conforme ele está coletando alimentos ou retornando à base.

### DefensiveAgent

A classe `DefensiveAgent` é uma especialização da classe `InferenceAgent` e representa um agente defensivo no jogo Pacman. Seu objetivo principal é proteger a base e impedir que os pacmans adversários coletem alimentos. No entanto, quando não há pacmans inimigos na metade do tabuleiro do agente, ele adota um comportamento ofensivo.

#### Funcionalidades

1. **`registerInitialState(self, gameState)`**
   - Chama o método `registerInitialState` da classe `InferenceAgent` para inicializar o estado do agente.
   - Inicializa a variável `offensing` para determinar se o agente deve adotar um comportamento ofensivo ou defensivo.

2. **`chooseAction(self, gameState)`**
   - Verifica se há pacmans inimigos na metade do tabuleiro do agente (`invaders`).
   - Obtém os tempos restantes de medo dos fantasmas adversários (`scaredTimes`).
   - Define o comportamento do agente:
     - Se não houver pacmans inimigos ou se os fantasmas inimigos estiverem assustados por um tempo significativo, o agente adota um comportamento ofensivo (`offensing = True`).
     - Caso contrário, o agente mantém um comportamento defensivo (`offensing = False`).
   - Chama o método `chooseAction` da classe `InferenceAgent` para selecionar a melhor ação com base nas crenças atualizadas sobre as posições dos inimigos.

3. **`evaluationFunction(self, gameState)`**
   - Avalia a utilidade de um estado do jogo para o agente defensivo.
   - Considera vários fatores, incluindo:
     - Posição atual do agente (`myPos`).
     - Distâncias prováveis aos inimigos (`enemyDistances`).
     - Distâncias aos pacmans inimigos (`pac_distances`) e aos fantasmas inimigos (`ghost_distances`).
     - Alimentos disponíveis no tabuleiro (`targetFood`) e suas distâncias (`foodDistances`).
     - Distâncias às cápsulas de poder que o agente está defendendo (`capsulesDistances`).
   - Retorna uma pontuação baseada nesses fatores, ajustando o comportamento do agente conforme ele está defendendo a base ou adotando um comportamento ofensivo.

