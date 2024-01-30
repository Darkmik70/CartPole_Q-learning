#!/usr/bin/env python
from cart_pole import CartPole
from q_learning_agent import Q_learning

def main():
    gamma = 0.7 # Discount rate
    alpha = 0.1 # Learning rate
    epsilon = 0.5 # How much we want to explore 
    episodes = 40_000 # Number of episodes

    isLearning = True # Set to False to test the trained model

    cart_pole = CartPole(isLearning)
    agent = Q_learning(cart_pole, gamma, alpha, epsilon, episodes, isLearning)
    agent.apply()

if __name__ == "__main__":
    main()
