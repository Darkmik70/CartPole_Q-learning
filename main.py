import random
import gymnasium as gym
import numpy as np
import sys
import matplotlib.pyplot as plt

STEPS = 6

class cartPoleEnv:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.current_state = self.env.reset()[0] #Returns tuple(2)
        self.isTerminated = False
        
    def reset_env(self):
        self.current_state = self.env.reset()[0]
        self.isTerminated = False
    def do_action(self,action):
       state, reward, self.isTerminated, _, _,= self.env.step(action)

       self.current_state = state #Update the state

       return self.discretize_states(state), reward

    def get_current_state(self):
        return self.discretize_states(self.current_state)
    
    def discretize_states(self, state):
        low_state_space = self.env.observation_space.low
        high_state_space = self.env.observation_space.high

        cartPositionBin = np.linspace(low_state_space[0], high_state_space[0], STEPS)
        cartVelocityBin = np.linspace(-4, 4, STEPS)
        poleAngleBin = np.linspace(low_state_space[2], high_state_space[2], STEPS)
        poleAngleVelocityBin = np.linspace(-4, 4, STEPS)

        indexPosition = np.maximum(np.digitize(state[0], cartPositionBin)-1, 0)
        indexVelocity = np.maximum(np.digitize(state[1], cartVelocityBin)-1, 0)
        indexAngle = np.maximum(np.digitize(state[2], poleAngleBin)-1, 0)
        indexAngularVelocity = np.maximum(np.digitize(state[3], poleAngleVelocityBin)-1, 0)

        return tuple([indexPosition, indexVelocity, indexAngle, indexAngularVelocity])

    def get_action_space(self):
        return self.env.action_space.n
    
    def is_task_done(self):
        return self.isTerminated
    

class Q_learning:
    def __init__(self, env, gamma, alpha, epsilon, episodes):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.episodes = episodes

        # Huge Q_Table
        self.Q_table = np.random.uniform(low =0, high = 1, size = (STEPS, STEPS, STEPS, STEPS, self.env.get_action_space()))

    def policy(self, state):
        """Epsilon Greedy Policy"""
        if np.random.random() < self.epsilon:
            # Choose an action at random with probability epsilon
            return random.choice([0,1]) # CartPole - we only have two actions
        else:
            # Choose the best action accordin to Q_table with probability 1-epsilon
            return np.argmax(self.Q_table[state])

    def apply(self):
        total_episode_rewards = []
        for e in range(self.episodes):
            
            # Epsilon decay
            # self.epsilon = max(0.01, 0.995 * self.epsilon)

            steps = 0
            episode_rewards = [] # rewards for each episode
            while not self.env.is_task_done():
                current_state = self.env.get_current_state()
                action = self.policy(current_state)
                next_state, reward = self.env.do_action(action)
        

                # Choose maximum Q-value for next state
                max_next_value = np.max(self.Q_table[next_state+(action,)])

                # Temporal difference update
                td_target = reward + self.gamma * max_next_value
                td_error = td_target - self.Q_table[current_state+(action,)]
                
                self.Q_table[current_state+(action,)] += self.alpha * td_error
                
                # Get episode 
                episode_rewards.append(reward)

            # Reset before new episode
            self.env.reset_env()
            # Update rewards
            print("Sum of rewards {}".format(np.sum(episode_rewards)))

            total_episode_rewards.append(np.sum(episode_rewards))
        # Calculate the mean
        print("Average reward after all episodes: ", np.mean(total_episode_rewards))
        plt.plot(total_episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Performance of Q-learning Agent')
        plt.show()



def main():
    gamma = 0.85 # Discount rate
    alpha = 0.2 # Learning rate
    epsilon = 0.9 # How much we want to explore
    episodes = 2500

    cart_pole = cartPoleEnv()
    agent = Q_learning(cart_pole, gamma, alpha, epsilon, episodes)
    agent.apply()


if __name__ == "__main__":
    main()
