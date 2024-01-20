import gymnasium as gym
import numpy as np


class CartPole:
    """
        Wrapper class for CartPole environment

        Attributes:
            _env: The Gym environment for the Cart Pole game.
            _curr_state (np.array): The current state of the environment.
            _isTerminated (bool): Flag indicating whether the current episode has ended.
    """
    def __init__(self, is_learning = False):
        """
        Initializes the CartPole environment

        Args:
            is_learning (bool): Flag to determine if the environment is for learning or visualization.
        """
        # Define whether we want to visualize
        if is_learning:
            self._env = gym.make('CartPole-v1')
        else:
            self._env = gym.make('CartPole-v1', render_mode = "human")
        self._currState = self._env.reset()[0]
        self._isTerminated = False


    def digitize_state(self, state):
        """
        Digitizes the continuous state into discrete values for Q-table.
        
        Args:
            state (np.array): The current state of the environment.

        Returns:
            list: A list representing the digitized state.
        """
        pos_space = np.linspace(-2.4, 2.4, 10)
        vel_space = np.linspace(-4, 4, 10)
        ang_space = np.linspace(-.2095, .2095, 10)
        ang_vel_space = np.linspace(-4, 4, 10)
        
        new_state_p = np.digitize(state[0], pos_space)
        new_state_v = np.digitize(state[1], vel_space)
        new_state_a = np.digitize(state[2], ang_space)
        new_state_av= np.digitize(state[3], ang_vel_space)
        new_state_dig = [new_state_p, new_state_v, new_state_a, new_state_av]
        return new_state_dig

    def do_action(self, action):
       """
        Performs a step in the environment. Gets the values for Observation, reward and checks if the game is over

        Args:
            action (int): an action passed to the environment
        Returns:
            new_state: Discrete state after the action is taken
            reward: Reward basing on the taken action
       """
       new_state, reward, self._isTerminated, _, _ = self._env.step(action)       
       # Update the current state
       self._currState = new_state
       return self.digitize_state(new_state), reward
    
    def reset_env(self):
        """ Resets the environment """
        self._currState = self._env.reset()[0]
        self._isTerminated = False

    def get_current_state(self):
        """ Gets the discrete state of the environment """
        return self.digitize_state(self._currState)
    
    def get_action_space(self):
        """Returns the size of the action space"""
        return self._env.action_space.n
    
    def is_game_over(self):
        """ Returns boolean determining if game is over"""
        return self._isTerminated