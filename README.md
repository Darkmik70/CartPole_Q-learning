# CartPole - Q-Learning
This project is a solution to the assignment given during AI For Robotics I course at UniGe.
The project number is 16.

# Project Structure
The project consists of two 3 components:
- `CartPole`: A wrapper class for OpenAI Gymnasium CartPole environment
- `Q_learnign`: Agent which implements Qlearnig algorithm

- `main.py`: Main function with all the variables

# Requirements
To successfully run the project few dependencies are needed.

- Python 3 (Possibly works with previous versions)
- gymnasium (OpenAI environment)
- numpy
- pickle (for saving and loading the Q-Table)


# Playing with the Agent
To run the project all you need is to run the `main.py`

```
python3 main.py
```

This will launch game with the predefined variables as well as in the learning mode. 

To switch on the mode into visualization you need change the hardcoded values in `main.py`

```python
    gamma = 0.7 # Discount rate
    alpha = 0.1 # Learning rate
    epsilon = 0.8 # How much we want to explore 
    episodes = 10000 # Number of episodes

    isLearning = True # Set to False to test the trained model
```

# Discussion
According to Author's observation the model requires a huge amount of episodes to be able to learn which decision to take depending on situation. 

The provided environment gives us observation space of 6 continous variables, which need to be digitized for the computational purpose. This creates a enormously huge (in authors opinion) Q_table, that requires at least *30k* episodes to improve the results 

In a few test runs shown that with Epsilon value being *0.5* the model is able to learn. However other set of variables might improve even further. 

Additionally the algorithm can use the Epsilon decay rate along with Epsilon Greedy Policy to reduce the randomness of choice along the operation, which can also infuence the efficiency of learning.