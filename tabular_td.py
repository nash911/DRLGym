import gym.spaces
import numpy as np
import sys, getopt
import matplotlib.pyplot as plt

def epsilon_greedy(env, Q, state, epsilon):
    # Choose an action at random
    action = env.action_space.sample()
    return action

def change_epsilon(episode, epsilon, epsilon_decay):
    return epsilon

def plot_learning_curve(history, l_algo):
    history = np.array(history)
    window_size = 200
    x = history[window_size-1:, 0]
    avg_g = np.convolve(history[:, 2], np.ones((window_size))/window_size,
                        mode='valid')

    plt.plot(x, avg_g, 'r-', label='Avg. Return')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Learning Curve (' + l_algo + ')')
    plt.grid(True)
    plt.savefig("learning_curve_" + l_algo + ".png")
    plt.show()

def main(argv):
    task = 'FrozenLake8x8-v0'
    l_algo = 'SARSA'
    num_episodes = 20001
    alpha = 1.0
    gamma = 0.1
    epsilon = 0.0
    epsilon_decay = 1.0
    render = False

    try:
        opts, args = getopt.getopt(argv,"hrt:l:n:a:g:e:d:",
            ["render", "task=" "learning_algorithm=", "num_episodes=", "alpha=",
             "gamma=", "epsilon=", "decay="])
    except getopt.GetoptError:
        print("tabular_rl.py -t <task> -l <learning_algorithm> -n <num_episodes>"
              " -a <alpha> -g <gamma> -e <epsilon> -d <decay> -r")
        sys.exit(2)

    for opt, arg in opts:
       if opt == '-h':
           print("tabular_rl.py -t <task> -l <learning_algorithm> -n <num_episodes>"
                 " -a <alpha> -g <gamma> -e <epsilon> -d <decay> -r")
           sys.exit()
       elif opt in ("-r", "--render"):
           render = True
       elif opt in ("-l", "--learning_algo"):
           if str(arg).lower() == 'sarsa':
               l_algo = 'SARSA'
           elif str(arg).lower() == 'q-learning':
               l_algo = 'Q-Learning'
           else:
               print("Incorrect argument for -l <learning_algorithm>: ", str(arg),
                     "\nShould be one of: {SARSA, Q-Learning}")
               sys.exit()
       elif opt in ("-t", "--task"):
           task = str(arg)
       elif opt in ("-n", "--num_episodes"):
           num_episodes = int(arg)
       elif opt in ("-a", "--alpha"):
           alpha = float(arg)
       elif opt in ("-g", "--gamma"):
           gamma = float(arg)
       elif opt in ("-e", "--epsilon"):
           epsilon = float(arg)
       elif opt in ("-d", "--decay"):
           epsilon_decay = float(arg)

    # Set the gym environment for the learning task
    env = gym.envs.make(task)

    # Reset environment and observe the initial state
    state = env.reset()

    # Initialize Q(s, a) = 0, ∀ s ∈ S, a ∈ A
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    # Create a list to store info about learning
    history = list()

    for episode in range(1, num_episodes):
        done = False
        steps = 0
        G = 0
        reward = 0

        # Reset the environment and observe the initial state
        state = env.reset()

        # Choose an action (a)
        action = epsilon_greedy(env, Q, state, epsilon)

        while done != True:
            # Execute the action and observe the resulting next state (s') and reward (r)
            state_prime, reward, done, info = env.step(action)

            # Choose the next action (a')
            action_prime = epsilon_greedy(env, Q, state_prime, epsilon)

            # Update Q-Value
            if l_algo == 'SARSA':
                # SARSA Update
                pass
            elif l_algo == 'Q-Learning':
                # Q-Learning Update
                pass

            #-----Total Discounted Return-----#
            #       T
            # G_T = ∑  ɣ^(T-t)r_t
            #      t=0
            G = reward + (gamma * G)

            # Set next state and action

            steps += 1
            if render:
                env.render()

        # Save info of the episode as history
        history.append([episode, steps, G, reward])

        # Decay epsilon
        epsilon = change_epsilon(episode, epsilon, epsilon_decay)

        if episode % 100 == 0:
            print('Episode {} - Steps {} - Epsilon {} - Goal: {} - Reward: '\
                  '{}'.format(episode, steps, epsilon, G, reward))

    # Plot Learning-Curve
    plot_learning_curve(history, l_algo)

if __name__ == "__main__":
    main(sys.argv[1:])
