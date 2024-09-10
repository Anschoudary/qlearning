#Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

#Class grid environment
class GridEnvironment:
    def __init__(self):
        self.grid_size = 5
        self.start_state = (2, 1)  #Agent starting position [2,1] in 1-indexed grid
        self.terminal_state = (5, 5)  #Termination state at [5,5] in 1-indexed grid
        self.jump_from = (2, 4)  #Special jump from position [2,4]
        self.jump_to = (4, 4)  #Special jump to position [4,4]
        self.obstacles = [(3, 5), (3, 3), (3, 4), (4, 3)]  #Obstacles positions in 1-indexed grid
        self.state_reset()

    def state_reset(self):  #Reset the state of agent to starting position (2, 1)
        self.state = self.start_state
        return self.state

    def move(self, action):
        x, y = self.state

        #Mapping the action to the correct direction
        if action == 1:  # North
            next_state = (max(x - 1, 1), y)
        elif action == 2:  # South
            next_state = (min(x + 1, self.grid_size), y)
        elif action == 3:  # East
            next_state = (x, min(y + 1, self.grid_size))
        elif action == 4:  # West
            next_state = (x, max(y - 1, 1))

        #Checking for obstacles
        if next_state in self.obstacles:
            next_state = self.state  #Agent will stay in the same position if it hits to an obstacle

        #Checking if the agent reaches the termination state
        if next_state == self.terminal_state:
            reward = 10
            done = True
        elif next_state == self.jump_from:  #Special jump treatment
            reward = 5
            next_state = self.jump_to
            done = False
        else:
            reward = -1
            done = False

        self.state = next_state
        return next_state, reward, done

    def is_terminal(self):
        return self.state == self.terminal_state

#Class Q-learning for agent
class QLAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((env.grid_size + 1, env.grid_size + 1, 4))  # 4 possible actions: North, South, East, West

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(1, 4)  #Choosing random action (1 to 4)
        else:
            x, y = state
            return np.argmax(self.q_table[x, y]) + 1  #Adding +1 to match action index with the correct move

    def learning(self, state, action, reward, next_state):
        x, y = state
        nx, ny = next_state
        predict = self.q_table[x, y, action - 1]  #action - 1 to map to 0-indexed array
        target = reward + self.gamma * np.max(self.q_table[nx, ny])
        self.q_table[x, y, action - 1] += self.alpha * (target - predict)

    def training(self, episodes):
        rewards = []
        for episode in range(episodes):
            state = self.env.state_reset()
            total_reward = 0
            while True:
                action = self.choose_action(state)
                next_state, reward, done = self.env.move(action)
                self.learning(state, action, reward, next_state)
                #Visualizing the agent state after every move
                visualize_grid_state(self, state)
                state = next_state
                total_reward += reward

                if done:
                    break
            rewards.append(total_reward)
            if len(rewards) > 30 and np.mean(rewards[-30:]) > 10:
                break
        return rewards

#Visualizing the state values and the grid
def visualize_grid_state(agent, current_state):
    
    plt.ion()  # Turn on interactive mode
    grid_size = agent.env.grid_size
    q_values = np.max(agent.q_table[1:, 1:], axis=2)

    plt.clf()  #Clearing the current figure
    ax = sns.heatmap(q_values, cmap="YlGnBu", cbar=True, square=True, annot=True,
                     xticklabels=range(1, grid_size + 1), yticklabels=range(1, grid_size + 1))

    #Drawing the obstacles
    for obstacle in agent.env.obstacles:
        ax.add_patch(
            plt.Rectangle((obstacle[1] - 1, obstacle[0] - 1), 1, 1, fill=True, color='black'))

    #Drawing the starting position
    ax.add_patch(
        plt.Rectangle((agent.env.start_state[1] - 1, agent.env.start_state[0] - 1), 1, 1, fill=True, color='green'))

    #Drawing the terminal position
    ax.add_patch(
        plt.Rectangle((agent.env.terminal_state[1] - 1, agent.env.terminal_state[0] - 1), 1, 1, fill=True, color='blue'))

    #Drawing the special jump positions
    ax.add_patch(
        plt.Rectangle((agent.env.jump_from[1] - 1, agent.env.jump_from[0] - 1), 1, 1, fill=True, color='orange'))
    ax.add_patch(plt.Rectangle((agent.env.jump_to[1] - 1, agent.env.jump_to[0] - 1), 1, 1, fill=True, color='yellow'))

    #Drawing the current agent position
    if current_state:
        ax.add_patch(plt.Circle((current_state[1] - 0.5, current_state[0] - 0.5), 0.2, color='red'))
        
    #Annotating the termination state with '+10'
    ax.text(agent.env.terminal_state[1] - 0.5, agent.env.terminal_state[0] - 0.5, '+10', color='white',
            ha='center', va='center', fontsize=12, fontweight='bold')

    plt.title("Q-values Heatmap with Grid World Layout")
    plt.draw()  #Redrawing the current figure
    plt.pause(0.1)  #Pausing for a short time to see the agent's movement

#Main function
def main():
    env = GridEnvironment()
    agent = QLAgent(env, alpha=1.0, gamma=0.9, epsilon=0.1)
    agent.training(100)

if __name__ == "__main__":
    main()