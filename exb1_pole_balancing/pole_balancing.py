import math
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# step snapshots pro episode memory
step_snapshot = namedtuple('StepSnapshot',
                           ('state', 'action', 'next_state', 'reward', 'terminated'))


# pamet pro ukladani a vzorkovani
class EpisodeMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(step_snapshot(*args))

    def sample(self, batch_size):
        # random sampling dat
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# neuronova sit pro aproximaci q funkce
# vstup: stav 's' (4 hodnoty)
# vystup: q hodnota pro kazdou akci stavu 's' (2 hodnoty - doleva, doprava)
class QNetwork(nn.Module):

    def __init__(self, n_observations, n_actions, hidden_size=256):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        # forward pass
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# agent ktery je ucen, Q learning + NN
class DQNAgent:
    def __init__(self,
                 n_observations,
                 n_actions,
                 episode_memory_size=10000,
                 epsilon_init=0.9,  # pocatecni hodnota epsilon (epsilon = pravdepodobnost explorace vs exploitace)
                 epsilon_final=0.01,  # konecna hodnota epsilon
                 epsilon_decay=1000,  # rychlost poklesu epsilon ve krocich
                 learning_rate=0.001,  # learning rate
                 batch_size=128,  # velikost batch pro trenink
                 ):

        self.device = torch.device("cpu")
        self.episode_memory_size = episode_memory_size

        # epsilon zacina vysoke pro exploraci
        self.epsilon_init = epsilon_init
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.n_actions = n_actions
        # NN pro predikci Q hodnot (exploitace). neustale aktualizovana
        self.online_net = QNetwork(n_observations, n_actions).to(self.device)

        # NN pro vypocet target hodnot - ... max_a'(Q(s', a')). Je zvlast pro stabilizaci treninku
        # aktualizovana kazdych target_network_update_count kroku
        self.target_net = QNetwork(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        self.memory = EpisodeMemory(self.episode_memory_size)
        self.steps_done = 0

    def calculate_epsilon(self):
        # vypocet aktualni hodnoty epsilon pro epsilon-greedy strategii
        # epsilon klesa exponencialne z epsilon_init na epsilon_final behem epsilon_decay kroku
        return self.epsilon_final + (self.epsilon_init - self.epsilon_final) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)

    # vyber akce pomoci epsilon greedy strategie
    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.calculate_epsilon()
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # vybrana akce s nejvyssi q-hodnotou
                return self.online_net(state).max(1)[1].view(1, 1)
        else:
            # nahodny vyber akce
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def learn(self):
        # treninkovy krok agenta
        if len(self.memory) < self.batch_size:
            return

        # ziskani batch z pameti
        transitions = self.memory.sample(self.batch_size)
        batch = step_snapshot(*zip(*transitions))

        # maska neterminalnich stavu a spojeni jejich tensoru
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)

        non_final_next_states_list = [s for s in batch.next_state if s is not None]
        if len(non_final_next_states_list) > 0:
            non_final_next_states = torch.cat(non_final_next_states_list).to(self.device)
        else:  # pokud jsou vsechny stavy v batch terminalni
            non_final_next_states = torch.empty(0, self.online_net.layer1.in_features, device=self.device)

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        terminated_batch = torch.tensor(batch.terminated, device=self.device, dtype=torch.float32)

        # vypocet q(s_t, a) - model predikuje q hodnoty a jsou vybrany ty pro akce, ktere byly provedeny
        state_action_values = self.online_net(state_batch).gather(1, action_batch)

        # vypocet v(s_{t+1}) pro vsechny nasledujici stavy
        # vysledna hodnota je 0 pokud byl stav terminalni
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if len(non_final_next_states_list) > 0:
            with torch.no_grad():
                # target net pro stabilizaci
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        # vypocet target q hodnot
        expected_state_action_values = (next_state_values * 0.99 * (1 - terminated_batch)) + reward_batch.squeeze(1)

        # vypocet loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # optimalizace modelu
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.online_net.parameters(), 100)
        self.optimizer.step()

    def update_target_net(self):
        # aktualizace vah target site vahami z online site
        self.target_net.load_state_dict(self.online_net.state_dict())
