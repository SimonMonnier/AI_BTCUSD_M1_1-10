import numpy as np
import pandas as pd
import random
from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque

import wandb

import torch
print("Version de PyTorch :", torch.__version__)
print("CUDA Disponible :", torch.cuda.is_available())
print("Nombre de GPUs :", torch.cuda.device_count())
print("Nom du GPU :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Aucun GPU")

# Vérification de la disponibilité du GPU
# Vérification de la disponibilité du GPU
device = torch.device("cuda")


# Initialisation de Weights & Biases
wandb.init(
    project="dqn ichimoku 2",
    config={
        "state_size": 26 * 13,  # Mise à jour de la taille de l'état
        "action_size": 4,
        "gamma": 0.95,
        "learning_rate": 1e-4,
        "memory_size": 2000,
        "batch_size": 32,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.0995,
        "episodes": 100
    }
)

# Paramètres de l'agent
EPISODES = 500
STATE_SIZE = 26 * 11 + 1# Mise à jour de la taille de l'état
ACTION_SIZE = 4  # Acheter, Vendre, Attendre, Clôturer
GAMMA = 0.95  # Facteur de réduction
LEARNING_RATE = 1e-4
MEMORY_SIZE = 12000
BATCH_SIZE = 32

class TradingEnvironment:
    def __init__(self, data):
        self.data = data.reset_index(drop=True)
        # Les données incluent désormais les indicateurs Ichimoku pré-calculés
        self.n_steps = len(self.data)
        self.current_step = 0
        self.position_buy = None  # None, 'buy', 'sell'
        self.position_sell = None  # None, 'buy', 'sell'
        self.entry_price_sell = 0.0
        self.entry_price_buy = 0.0
        self.lot = 0.01
        self.wait_buy = 0
        self.wait_sell = 0
        self.previous_action = 0
        self.profit_sell = 0.0
        self.profit_buy = 0.0
        self.drawdown = 0.0
        self.drawup = 0.0
        self.previous_profit_buy = 0.0
        self.previous_profit_sell = 0.0

        # Pré-calculer les états pour optimiser les performances
        self.start_index = 52  # Commencer après les 52 premières bougies pour avoir assez de données
        self.end_index = self.n_steps - 26  # Pour éviter de dépasser lors de l'accès aux valeurs futures
        self.states = []

        for current_step in range(self.start_index, self.end_index):
            state = []
            for i in range(current_step - 26, current_step):
                data_i = self.data.iloc[i]
                # Obtenir les valeurs futures de Senkou Span A et B (nuage Kumo)
                # Obtenir les valeurs passées de Senkou Span A et B (26 périodes dans le passé)
                if i - 26 >= 0:
                    past_senkou_span_a = self.data.iloc[i - 26]['senkou_span_a']
                    past_senkou_span_b = self.data.iloc[i - 26]['senkou_span_b']
                else:
                    past_senkou_span_a = 0.0
                    past_senkou_span_b = 0.0
                state_i = np.concatenate([
                    data_i[['open', 'high', 'low', 'close', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']].astype(float).values,
                    [float(past_senkou_span_a), float(past_senkou_span_b)]
                ])
                state.append(state_i)
            state = np.array(state).flatten()
            self.states.append(state)

    def reset(self):
        self.closed_trades = []
        self.current_step = 0
        self.position_buy = None
        self.position_sell = None
        self.entry_price_buy = 0.0
        self.entry_price_sell = 0.0
        self.previous_action = 0  # Réinitialiser l'action précédente
        self.profit_sell = 0.0
        self.profit_buy = 0.0
        self.drawdown = 0.0
        self.drawup = 0.0
        self.previous_profit_buy = 0.0
        self.previous_profit_sell = 0.0
        state = self.states[self.current_step]
        state = np.append(state, [0])
        return state

    def get_state(self):
        return self.states[self.current_step]

    def step(self, action):
        
        reward = 0
        done = False
        perte = 0
        profit = 0
        futur_price = 0
        real_step = self.current_step + self.start_index  # Obtenir l'indice réel dans self.data
        current_price = self.data.iloc[real_step]['close']
        current_price_high = self.data.iloc[real_step]['high']  # Prix de clôture actuel
        current_price_low = self.data.iloc[real_step]['low']
        if self.current_step+1 >= len(self.states):
            done = True
        else:
            futur_price = self.data.iloc[real_step+1]['close']  # Prix de clôture future

        # Calcul des profits courants
        profit_buy = 0
        profit_sell = 0

        # Calcul des profits courants
        if self.position_buy == 'buy':
            profit_buy_tp = current_price_high - self.entry_price_buy
            profit_buy_sl = current_price_low - self.entry_price_buy
            if profit_buy_tp >= 300 :
                self.position_buy = None
                reward = 3.14
                profit_buy = 300
                self.closed_trades.append(profit_buy)
            elif profit_buy_sl <= -30 :
                self.position_buy = None
                reward = -0.314
                profit_buy = -30
                self.closed_trades.append(profit_buy)

        if self.position_sell == 'sell':
            profit_buy_tp = current_price_low - self.entry_price_buy
            profit_buy_sl = current_price_high - self.entry_price_buy
            if profit_sell >= 300 :
                self.position_sell = None
                reward = 3.14
                profit_sell = 300
                self.closed_trades.append(profit_sell)
            elif profit_sell <= -30 :
                self.position_sell = None
                reward = -0.314
                profit_sell = -30
                self.closed_trades.append(profit_sell)
            else:
                profit_sell = 0

        # Actions: 0 - Attendre, 1 - Acheter, 2 - Vendre, 3 - Clôturer
        if action == 1:  # Acheter
            
            if self.position_buy is None:
                self.entry_price_buy = current_price
                self.position_buy = 'buy'
                if futur_price > current_price:
                    reward += 0.1618033
            else:
                reward += -0.001618033

        elif action == 2:  # Vendre
            if self.position_sell is None:
                self.entry_price_sell = current_price
                self.position_sell = 'sell'
                if futur_price < current_price:
                    reward += 0.1618033
            else:
                reward += -0.001618033

        if action == 0 or action == 3:  # Attendre
            reward -= 0.001618  # Pénalité pour l'inaction


        self.current_step += 1

        if self.current_step >= len(self.states):
            done = True

        if not done:
            next_state = self.states[self.current_step]
        else:
            next_state = np.zeros(STATE_SIZE - 1)  # STATE_SIZE - 3 car nous ajouterons action et profits
                
        if profit_buy < 0:
            perte += profit_buy
        elif profit_buy > 0:
            profit += profit_buy
        if profit_sell < 0:
            perte += profit_sell
        elif profit_sell > 0:
            profit += profit_sell

        order_state = 0

        if self.position_buy == 'buy':
            order_state = 1
        if self.position_sell == 'sell':
            order_state = 2

        # Ajouter une petite récompense proportionnelle à la différence de profit
        reward = np.tanh(reward) * 0.01618033  # Ajustez le facteur d'échelle si nécessaire

        marge = perte + profit 
          # Mettre à jour l'action précédente
        next_state = np.append(next_state/100000, [order_state])
        #next_state = np.append(next_state, [self.previous_action, profit_sell, profit_buy, buy_state, sell_state])
        
        return next_state, reward, done, profit, perte, marge

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 469)
        self.fc2 = nn.Linear(469, 290)
        self.output = nn.Linear(290, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x

class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha  # Niveau de priorité (0 = uniforme, 1 = prioritaire)

    def add(self, experience, td_error):
        priority = (abs(td_error) + 1e-5) ** self.alpha  # Calcul de la priorité basée sur l'erreur TD
        self.buffer.append(experience)
        self.priorities.append(priority)


    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == len(self.priorities):
            priorities = np.array(self.priorities, dtype=np.float32)
            scaled_priorities = priorities ** self.alpha
            sampling_probabilities = scaled_priorities / scaled_priorities.sum()

            indices = np.random.choice(len(self.buffer), batch_size, p=sampling_probabilities)
            experiences = [self.buffer[idx] for idx in indices]

            # Calcul des poids d'importance sampling
            total = len(self.buffer)
            weights = (total * sampling_probabilities[indices]) ** (-beta)
            weights /= weights.max()
            weights = np.array(weights, dtype=np.float32)

            return experiences, indices, weights
        else:
            return [], [], []

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-5  # Éviter les priorités nulles

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.step_count = 0
        self.update_frequency = 618  # Mettre à jour le réseau cible tous les 1000 pas de temps (ajustez selon vos besoins)

        self.memory = PrioritizedReplayMemory(MEMORY_SIZE)
        self.gamma = GAMMA
        self.learning_rate = LEARNING_RATE
        self.tau_target = 0.001

        # Paramètres pour l'exploration Softmax
        self.tau_start = 1.0   # Valeur initiale de tau
        self.tau_end = 0.01    # Valeur minimale de tau
        self.tau = self.tau_start  # Initialisation de tau

        # Pour l'adaptation dynamique (optionnel)
        self.performance_window = deque(maxlen=100)
        self.previous_performance = None

        self.model = DQN(self.state_size, self.action_size).to(device)
        self.target_model = DQN(self.state_size, self.action_size).to(device)
        #self.update_target_network()
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5, betas=(0.9, 0.995))

        self.criterion = nn.SmoothL1Loss()

    def update_target_network(self):
        tau = self.tau_target
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def remember(self, state, action, reward, next_state, done):
        # Calcul initial de l'erreur TD pour définir la priorité
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        action_tensor = torch.LongTensor([action]).unsqueeze(0).to(device)
        reward_tensor = torch.FloatTensor([reward]).unsqueeze(0).to(device)
        done_tensor = torch.FloatTensor([done]).unsqueeze(0).to(device)

        with torch.no_grad():
            q_value = self.model(state_tensor).gather(1, action_tensor)
            next_action = self.model(next_state_tensor).argmax(1).unsqueeze(1)
            target_next_q_value = self.target_model(next_state_tensor).gather(1, next_action)
            target_q_value = reward_tensor + (1 - done_tensor) * self.gamma * target_next_q_value
            td_error = (q_value - target_q_value).cpu().numpy()[0][0]

        self.memory.add((state, action, reward, next_state, done), td_error)

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state)
        q_values = q_values.cpu().numpy()[0]
        
        # Normalisation des q-values
        mean_q = np.mean(q_values)
        std_q = np.std(q_values)
        if std_q > 0:
            q_values_normalized = (q_values - mean_q) / std_q
        else:
            q_values_normalized = q_values - mean_q
        
        # Application de la température pour l'exploration Softmax
        preferences = q_values_normalized / self.tau
        max_preference = np.max(preferences)
        exp_preferences = np.exp(preferences - max_preference)
        action_probs = exp_preferences / np.sum(exp_preferences)
        action = np.random.choice(self.action_size, p=action_probs)
        return action



    def replay(self, beta=0.4):
        if len(self.memory.buffer) < BATCH_SIZE:
            return

        experiences, indices, weights = self.memory.sample(BATCH_SIZE, beta)
        if not experiences:
            return

        # Conversion en tensors
        states = torch.FloatTensor(np.array([e[0] for e in experiences])).to(device)
        actions = torch.LongTensor(np.array([e[1] for e in experiences])).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(np.array([e[2] for e in experiences])).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array([e[3] for e in experiences])).to(device)
        dones = torch.FloatTensor(np.array([e[4] for e in experiences])).unsqueeze(1).to(device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(device)

        # Calcul des valeurs Q
        q_values = self.model(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1).unsqueeze(1)
            target_next_q_values = self.target_model(next_states).gather(1, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * target_next_q_values

        # Calcul des erreurs TD
        td_errors = (q_values - target_q_values).detach().cpu().numpy().flatten()

        # Nouvelle fonction de perte
        loss = (weights * self.criterion(q_values, target_q_values)).mean()

        # Optimisation du modèle
        self.optimizer.zero_grad()
        loss.backward()

        # === Ajouter le clipping des gradients ici ===
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Incrémenter le compteur de pas de temps
        self.step_count += 1

        # Mettre à jour le réseau cible tous les C pas de temps
        if self.step_count % self.update_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        # Mise à jour des priorités
        self.memory.update_priorities(indices, td_errors)

        return loss.item()

def train_agent(csv_file, episodes=2000):
    # Chargement des données
    data = pd.read_csv(csv_file)

    # Calcul des composants Ichimoku
    high = data['high']
    low = data['low']
    close = data['close']

    # Tenkan-sen (Ligne de conversion)
    period9_high = high.rolling(window=9).max()
    period9_low = low.rolling(window=9).min()
    data['tenkan_sen'] = (period9_high + period9_low) / 2

    # Kijun-sen (Ligne de base)
    period26_high = high.rolling(window=26).max()
    period26_low = low.rolling(window=26).min()
    data['kijun_sen'] = (period26_high + period26_low) / 2

    # Senkou Span A (Ligne de tendance 1)
    data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(26)

    # Senkou Span B (Ligne de tendance 2)
    period52_high = high.rolling(window=52).max()
    period52_low = low.rolling(window=52).min()
    data['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)

    # Chikou Span (Ligne retardée)
    data['chikou_span'] = close.shift(-26)

    # Supprimer les lignes avec des valeurs manquantes
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)

    env = TradingEnvironment(data)
    agent = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)
    # agent.model.load_state_dict(torch.load("dqn_trading_model.pth"))
    rewards_per_episode = []

    beta_start = 0.4

    #agent.model.load_state_dict(torch.load('dqn_trading_model.pth'))
    #agent.model.eval()  # Passe le modèle en mode évaluation
    marge_global = 0
    best_rewards = -999999999999999
    best_marge = -999999999999999

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        total_profit = 0
        total_perte = 0
        total_marge = 0
        done = False
        #torch.save(agent.model.state_dict(), "dqn_trading_model.pth")
        # Pour le logging des actions
        actions_count = {0: 0, 1: 0, 2: 0, 3: 0}

        # Calcul de beta pour le PER
        beta = min(1.0, beta_start + (1.0 - beta_start) * (e / (episodes - 1)))

        losses = []

        while not done:
            action = agent.act(state)
            actions_count[action] += 1  # Comptabiliser l'action choisie
            next_state, reward, done, profit, perte, marge = env.step(action)
            total_reward += reward
            total_profit += profit
            total_perte += perte
            total_marge += marge
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            loss = agent.replay(beta=beta)  # Passer beta à replay()

            if loss is not None:
                losses.append(loss)
            else:
                losses.append(0)

        average_loss = sum(losses) / len(losses)

        # Réduction de l'exploration
        # Mise à jour du paramètre de température tau en fonction des épisodes
        #agent.tau = agent.tau_end + (agent.tau_start - agent.tau_end) * (1 - e / (EPISODES - 1))
        tau = agent.tau_start - ((agent.tau_start - agent.tau_end) * e / (EPISODES*0.75))
        agent.tau = max(tau, agent.tau_end)

                
        rewards_per_episode.append(total_reward)

        # Mise à jour du réseau cible
        #agent.update_target_network()

        marge_global += total_marge

        if best_rewards < total_reward:
            torch.save(agent.model.state_dict(), "dqn_trading_model_best_rewards_2-1.pth")
            best_rewards = total_reward
            print("Saving best rewards model dqn_trading_model_best_rewards_2-1.pth")
        if best_marge < total_marge:
            torch.save(agent.model.state_dict(), "dqn_trading_model_best_marge_2-1.pth")
            best_marge = total_marge
            print("Saving best marge model dqn_trading_model_best_marge_2-1.pth")

        # À la fin de chaque épisode
        closed_trades = env.closed_trades

        if closed_trades:
            # Initialisation des compteurs
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            current_consecutive_wins = 0
            current_consecutive_losses = 0
            max_consecutive_gain = 0
            max_consecutive_loss = 0
            cumulative_gain = 0
            cumulative_loss = 0
            num_winning_trades = 0
            num_losing_trades = 0

            for profit in closed_trades:
                if profit > 0:
                    num_winning_trades += 1
                    current_consecutive_wins += 1
                    cumulative_gain += profit
                    current_consecutive_losses = 0
                    cumulative_loss = 0  # Réinitialiser la perte cumulative

                    # Mettre à jour le nombre maximal de trades gagnants consécutifs
                    if current_consecutive_wins > max_consecutive_wins:
                        max_consecutive_wins = current_consecutive_wins
                    # Mettre à jour la hausse maximale de gain consécutif
                    if cumulative_gain > max_consecutive_gain:
                        max_consecutive_gain = cumulative_gain
                else:
                    num_losing_trades += 1
                    current_consecutive_losses += 1
                    cumulative_loss += profit  # Notez que profit est négatif
                    current_consecutive_wins = 0
                    cumulative_gain = 0  # Réinitialiser le gain cumulatif

                    # Mettre à jour le nombre maximal de trades perdants consécutifs
                    if current_consecutive_losses > max_consecutive_losses:
                        max_consecutive_losses = current_consecutive_losses
                    # Mettre à jour la chute maximale de perte consécutive
                    if cumulative_loss < max_consecutive_loss:
                        max_consecutive_loss = cumulative_loss  # cumulative_loss est négatif
        else:
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            max_consecutive_gain = 0
            max_consecutive_loss = 0
            num_winning_trades = 0
            num_losing_trades = 0


        # Enregistrement des métriques dans wandb
        wandb.log({
            "Total Reward": total_reward,
            "Total Profit": total_profit,
            "Total Perte": total_perte,
            "Total Marge": total_marge,
            "Tau": agent.tau,
            "Attendre buy": actions_count[0],
            "Attendre sell": actions_count[3],
            "Buy": actions_count[1],
            "Sell": actions_count[2],
            "Marge Globale": marge_global,
            "Average Loss": average_loss,
            "Actions Histogram": wandb.Histogram(list(actions_count.values())),
            "Consecutive Loss": max_consecutive_loss,
            "Consecutive Gain": max_consecutive_gain,
            "Consecutive trade Wins": max_consecutive_wins,
            "Consecutive trade Losses": max_consecutive_losses,
        })

        print(f"Épisode {e+1}/{episodes} - Récompense totale: {total_reward:.2f} - Tau: {agent.tau:.4f} - Beta: {beta:.4f}")

    # Sauvegarder le modèle
    torch.save(agent.model.state_dict(), "dqn_trading_model_last_rewards_2-1.pth")

    # Visualiser les récompenses
    plt.plot(rewards_per_episode)
    plt.xlabel('Épisode')
    plt.ylabel('Récompense Totale')
    plt.title('Performance de l\'Agent par Épisode')
    plt.show()

    return agent

if __name__ == "__main__":
    csv_file = 'BTCUSD_M1_data.csv'  # Remplacez par le chemin vers votre fichier CSV
    trained_agent = train_agent(csv_file, episodes=EPISODES)
