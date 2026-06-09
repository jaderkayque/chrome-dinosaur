# 🦖 Dino AI

Uma recriação do clássico **Chrome Dinosaur** (o jogo do dinossauro que aparece
no Google Chrome quando você fica sem internet), feita em **Python + Pygame** e
turbinada com **dois agentes de Inteligência Artificial** que aprendem a jogar
sozinhos: um por **Aprendizado por Reforço Profundo** e outro por
**Neuroevolução**.

Você pode jogar, assistir a IA jogar, treinar os agentes do zero ou até
**apostar uma corrida contra a IA**.

---

## 📑 Índice

- [O Jogo](#-o-jogo)
- [A Inteligência por Trás](#-a-inteligência-por-trás)
  - [O ambiente compartilhado (DinoEnv)](#o-ambiente-compartilhado-dinoenv)
  - [Agente 1 — D3QN (Deep Reinforcement Learning)](#agente-1--d3qn-deep-reinforcement-learning)
  - [Agente 2 — NEAT (Neuroevolução)](#agente-2--neat-neuroevolução)
- [Desempenho dos Agentes](#-desempenho-dos-agentes)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Instalação](#-instalação)
- [Como Usar](#-como-usar)
- [Controles](#-controles)
- [Bibliotecas](#-bibliotecas)

---

## 🎮 O Jogo

Um dinossauro corre por um deserto em velocidade **cada vez maior**, e precisa
desviar de dois tipos de obstáculo:

- 🌵 **Cactos** (pequenos e grandes) — exigem um **pulo**.
- 🐦 **Pássaros** — voam na altura do dinossauro e exigem um **agachamento**.

A cada 100 pontos o jogo acelera, até atingir a velocidade máxima. A pontuação é
simplesmente **quanto tempo você sobrevive**. Quanto mais rápido o jogo fica,
mais apertada fica a janela de reação — o que torna o desafio interessante tanto
para humanos quanto para algoritmos.

---

## 🧠 A Inteligência por Trás

O coração do projeto são **dois paradigmas de IA completamente diferentes**
aprendendo a mesma tarefa. Ambos enxergam o jogo através do **mesmo conjunto de
sensores** e dispõem das **mesmas três ações**, o que torna a comparação entre
eles justa e direta.

### O ambiente compartilhado (`DinoEnv`)

Os dois agentes interagem com o jogo por uma interface no estilo *gym*
(`reset()` / `step(ação)`), exatamente como em ambientes profissionais de
pesquisa em IA. A cada quadro, o agente recebe um **vetor de 9 números
normalizados** descrevendo a situação:

| # | Sensor | Significado |
|---|--------|-------------|
| 0 | `dist_x` | Distância horizontal até o próximo obstáculo |
| 1 | `width` | Largura do obstáculo |
| 2 | `height` | Altura do obstáculo |
| 3 | `speed` | Velocidade atual do jogo |
| 4 | `dino_y` | Posição vertical do dinossauro |
| 5 | `vel_y` | Velocidade vertical (subindo/descendo no pulo) |
| 6 | `is_bird` | Se o obstáculo é um pássaro |
| 7 | `gap` | Distância até o **segundo** obstáculo |
| 8 | `next_type` | Tipo do segundo obstáculo |

E escolhe uma de **três ações**: `CORRER`, `PULAR` ou `AGACHAR`.

Para guiar o aprendizado, o ambiente usa um sistema de **recompensas**: pequeno
bônus por sobreviver a cada quadro, bônus maior (proporcional à velocidade) por
passar um obstáculo, e uma penalidade forte ao colidir. Há ainda pequenas
penalidades por ações desperdiçadas (pular longe demais do obstáculo, agachar
sem necessidade), que ensinam o agente a agir com economia.

### Agente 1 — D3QN (Deep Reinforcement Learning)

O primeiro agente é uma **rede neural profunda** treinada por **Aprendizado por
Reforço**. Ele aprende, por tentativa e erro, a estimar o "valor" de cada ação
em cada situação. A arquitetura combina três técnicas modernas e consolidadas:

- **Dueling Network** — a rede separa *"quão boa é esta situação"* de *"qual ação
  é a melhor aqui"*, o que melhora a generalização.
- **Double DQN** — usa duas redes (uma escolhe a ação, outra a avalia) para
  evitar superestimar o valor das ações — um problema clássico do método.
- **Prioritized Experience Replay (PER)** — em vez de reaprender com experiências
  aleatórias, o agente **prioriza as situações mais surpreendentes** (onde mais
  errou), acelerando o aprendizado. Implementado com uma *SumTree* para
  amostragem eficiente.

Outros detalhes do treinamento incluem atualização suave da rede-alvo,
*frame skipping* (uma decisão é mantida por alguns quadros) e exploração
ε-greedy (começa explorando muito e vai ficando cada vez mais decidido).

> Resultado: uma política **muito consistente** — o agente raramente vacila e
> mantém um desempenho estável corrida após corrida.

### Agente 2 — NEAT (Neuroevolução)

O segundo agente não usa gradiente nem retropropagação. Em vez disso, ele
**evolui** uma população de redes neurais ao longo de gerações, como uma seleção
natural artificial — o algoritmo **NEAT** (*NeuroEvolution of Augmenting
Topologies*).

- Começa com **50 redes** competindo simultaneamente na mesma pista.
- As que sobrevivem mais tempo (maior *fitness*) se reproduzem; as piores são
  descartadas.
- A cada geração ocorrem **mutações** que alteram pesos e até a **própria
  estrutura da rede** (adicionando/removendo neurônios e conexões).
- Com o tempo, a evolução não só melhora o desempenho como também **enxuga** a
  rede, descartando conexões inúteis.

O resultado mais curioso é que a melhor rede evoluída costuma ser
**surpreendentemente simples e interpretável**: ela aprende sozinha a ignorar
sensores irrelevantes e a montar "circuitos" com função clara — por exemplo, um
neurônio dedicado a detectar pássaros e disparar o agachamento.

> Resultado: uma política de **alto desempenho de pico** — capaz de corridas
> extraordinariamente longas, embora com mais variação entre uma corrida e outra.

---

## 📊 Desempenho dos Agentes

Avaliação dos modelos treinados ao longo de 50 corridas (pontuação = quadros
sobrevividos):

| Métrica | D3QN | NEAT |
|---|---|---|
| Pontuação média | ~1.840 | **~4.870** |
| Mediana | ~1.780 | **~3.720** |
| Melhor corrida | ~3.130 | **~13.900** |
| Consistência (menor variação) | **Muito alta** | Moderada |

Em resumo: o **NEAT pontua mais** em média e atinge picos muito maiores, enquanto
o **D3QN é mais previsível e estável**. Dois caminhos diferentes, duas
personalidades diferentes de jogo.

---

## 📁 Estrutura do Projeto

```
chrome-dinosaur/
├── main.py              # Menu principal (jogar, replay, treinar)
├── core/                # Motor do jogo, compartilhado por todos
│   ├── game.py          #   DinoEnv: física, obstáculos, sensores, recompensas
│   ├── constants.py     #   Todos os parâmetros do jogo e dos agentes
│   ├── hud.py           #   Interface com gráficos e visualização da rede
│   └── versus.py        #   Modo Humano vs IA
├── dqn/                 # Agente de Aprendizado por Reforço
│   ├── model.py         #   Arquitetura Dueling
│   ├── agent.py         #   Double DQN + Prioritized Experience Replay
│   ├── train_dqn.py     #   Treinamento
│   └── replay_dqn.py    #   Assistir ao agente jogar
├── neat_ai/             # Agente de Neuroevolução
│   ├── config.txt       #   Configuração evolutiva do NEAT
│   ├── train_neat.py    #   Evolução das gerações
│   └── replay_neat.py   #   Assistir ao melhor genoma jogar
└── Assets/              # Sprites e imagens do jogo
```

---

## ⚙️ Instalação

Requisitos: **Python 3.10+**.

```bash
# 1. Crie e ative um ambiente virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 2. Instale as dependências
pip install pygame torch neat-python numpy
```

---

## 🚀 Como Usar

A forma mais simples é abrir o **menu principal**:

```bash
python main.py
```

A partir dele você pode **jogar contra a IA**, **assistir** a cada agente jogar
ou **treinar** do zero (ou continuar de onde parou).

Também é possível executar cada modo diretamente:

```bash
# Assistir aos agentes treinados
python dqn/replay_dqn.py
python neat_ai/replay_neat.py

# Treinar (rode a partir da raiz do projeto)
python dqn/train_dqn.py            # treina o D3QN do zero
python dqn/train_dqn.py --resume   # continua de onde parou
python neat_ai/train_neat.py       # evolui o NEAT
```

Durante o treino do D3QN, uma tecla permite ativar o **modo rápido** (sem
renderização), acelerando bastante o aprendizado.

---

## 🎯 Controles

**Jogando (modo Humano vs IA):**
- `↑` ou `Espaço` — pular
- `↓` — agachar
- `R` — reiniciar a corrida
- `ESC` — sair

**Durante treino / replay:**
- `ESC` — salvar e sair
- `F` — alternar modo rápido (treino do D3QN)
- `R` — reiniciar episódio (replay)

---

## 📚 Bibliotecas

- **[Pygame](https://www.pygame.org/)** — motor do jogo e renderização.
- **[PyTorch](https://pytorch.org/)** — rede neural e treinamento do D3QN.
- **[neat-python](https://neat-python.readthedocs.io/)** — algoritmo de
  neuroevolução NEAT.
- **[NumPy](https://numpy.org/)** — operações numéricas e vetores de estado.

---

## 🙏 Créditos

A base do jogo (sprites e mecânica original do Chrome Dinosaur em Pygame) foi
construída a partir do template de [Max Rohowsky](https://github.com/maxrohowsky/chrome-dinosaur).
As camadas de Inteligência Artificial (D3QN e NEAT), o ambiente `DinoEnv` e os
modos de treino/replay/versus são contribuições deste projeto.

---

Divirta-se jogando — e veja se você consegue bater a IA! 🦖
