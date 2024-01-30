import numpy as np
import random
import pickle


class Q_learning:
    """
        Implementation of Q-learning algorhitm for the CartPole environment.

        Attributes:
            _env (cartPoleEnv): Cart Pole env
            _gamma (float):   The discount factor
            _alpha (float): The learning rate.
            _epsilon (float): The exploration rate.
            _episodes (int): The number of episodes for training
            _is_learning (bool): Flag indicating whether the agent is in learning mode.
            _Q_table (np.array): The Q-table, stores state-action values
    """
    def __init__(self, env, gamma, alpha, epsilon, episodes, isLearning = True):
        """
            Initializes Q-learning agent.

            Works in two ways. When isLearning flag is set True,
            it initializes Q-table as a empty np.array, else it tries to load it from file.
            Args:
                env (cartPoleEnv): The Cart Pole environment.
                gamma (float): The discount factor.
                alpha (float): The learning rate.
                epsilon (float): The exploration rate.
                episodes (int): The number of episodes for training.
                isLearning (bool): Flag to determine if the agent is in learning mode.
        """
        self._env = env
        self._gamma = gamma
        self._alpha = alpha
        self._epsilon = epsilon
        self._episodes = episodes
        self._isLearning = isLearning
        self._decayRate = epsilon / episodes

        if self._isLearning:
            print(f'Learning mode on: training agent on alpha: {self._alpha}, gamma: {self._gamma}, epsilon : {self._epsilon}, with {self._episodes} episodes')
        else:
            print('Visualization mode on')

        # Initialize Q_Table
        if self._isLearning: 
            # State is given as continuous set of variables
            # we need to cut it into pieces to be able to learn
            # The limits here are the limits for our game to be over
            pos_space = np.linspace(-2.4, 2.4, 10)
            vel_space = np.linspace(-4, 4, 10)
            ang_space = np.linspace(-.2095, .2095, 10) #value in rad
            ang_vel_space = np.linspace(-4, 4, 10)
            self.Q_table = np.zeros((len(pos_space)+1, len(vel_space)+1, 
                                    len(ang_space)+1, len(ang_vel_space)+1, self._env.get_action_space())) #11x11x11x11x2
        else:
            #Load the model
            f = open('Q_table.pkl', 'rb')
            self.Q_table = pickle.load(f)
            f.close()

    def policy(self, state):
        """ 
        Epsilon Greedy Policy

        Function works in two modes:
            If isLearning is True, decides on random whether to choose random action or
            the best action according to the Q_table. The higher epsilon, the higher chance of getting random results
            When isLearning is set to False, policy only chooses the values basing on the Q_table.
        
        Args:
            state: Discrete state of the environment
        """
        if self._isLearning and np.random.random() < self._epsilon:
            # Choose an action at random with probability epsilon
            return random.choice([0,1]) # only two actions - left or right
        else:
            # Choose the best action accordin to Q_table with probability 1-epsilon
            return np.argmax(self.Q_table[state[0], state[1], state[2], state[3], :])

    def apply(self):
        """
        Executes Q-learning algorhithm over a specified number of episodes.

        This method runs the Q-learning algorithm, updating the Q-table based on the interactions
        with the environment. It implements an epsilon-greedy policy for action selection and applies 
        temporal difference learning for updating the Q-table.
        Additionally, the method also handles epsilon decay.
         
        For exploration over time and prints out the progress every 100 episodes.

        The method performs the following steps in each episode:
        - Interacts with the environment to obtain states, rewards, and new states.
        - Updates the Q-table using the temporal difference
        - Applies epsilon decay to gradually shift from exploration to exploitation.
        - Tracks and logs the rewards for each episode.

        At the end of the training, the updated Q-table is saved to a file (if in learning mode), 
        and the average reward across all episodes is calculated and printed to the output.
        """

        total_episode_rewards = []  # Rewards of all runs
        for episode in range(self._episodes):
            episode_rewards = [] # rewards for each episode
            rewards = 0
            while not self._env.is_game_over():
                # get the current state
                curr_state = self._env.get_current_state()
                action = self.policy(curr_state)
                next_state, reward = self._env.do_action(action)
                # Choose maximum Q-value for next state
                max_next_value = np.max(self.Q_table[next_state[0], next_state[1], next_state[2], next_state[3], :])
                # Temporal difference update TODO improve readability
                self.Q_table[curr_state[0], curr_state[1], curr_state[2], curr_state[3], action] = self.Q_table[curr_state[0], curr_state[1], curr_state[2], curr_state[3], action] +\
                self._alpha * ( reward + self._gamma * max_next_value -  self.Q_table[curr_state[0], curr_state[1], curr_state[2], curr_state[3], action]) 
                rewards += reward

            # Reset before new episode
            self._env.reset_env()

            # Epsilon Decay rate 
            self._epsilon = self._epsilon - self._decayRate
            
            # Get episode  rewards
            total_episode_rewards.append(rewards)
            mean_rewards = np.mean(total_episode_rewards[len(total_episode_rewards)-100:])
            
            if not self._isLearning:
                # Display results after each episode
                print(f'Episode: {episode} Rewards: {rewards}')
            else:
                # For every 100 display rewards
                if episode % 100 == 0:
                    print(f'Episode: {episode} Rewards: {rewards}  Epsilon: {self._epsilon:0.2f}  Mean Rewards {mean_rewards:0.1f}')
                    total_episode_rewards.append(np.sum(episode_rewards))
            
            # Threshold for rewards
            if mean_rewards >= 1000:
                print(f' Mean rewards: {mean_rewards} - no need to train model longer')
                break
        
        # Save Q table to file
        if self._isLearning:
            f = open('Q_table.pkl','wb')
            pickle.dump(self.Q_table, f)
            f.close()

        # Calculate the mean
        print("Average reward after all episodes: ", np.mean(total_episode_rewards))