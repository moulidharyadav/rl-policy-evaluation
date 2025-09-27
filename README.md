# POLICY EVALUATION

## AIM
To develop a Python program to evaluate the given policy by maximizing its cumulative reward while dealing with slippery terrain.

## PROBLEM STATEMENT
We are assigned with the task of creating an RL agent to solve the "Bandit Slippery Walk" problem. The environment consists of Seven states representing discrete positions the agent can occupy. The agent must learn to navigate this environment while dealing with the challenge of slippery terrain. Slippery terrain introduces stochasticity in the agent's actions, making it difficult to predict the outcomes of its actions accurately.

## POLICY EVALUATION FUNCTION
<img width="685" height="130" alt="image" src="https://github.com/user-attachments/assets/bb7f8627-3108-4d08-a052-de7522e75645" />

## PROGRAM
```
pip install git+https://github.com/mimoralea/gym-walk#egg=gym-walk


import warnings ; warnings.filterwarnings('ignore')

import gym
import numpy as np

import random
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True)
random.seed(123); np.random.seed(123);


def print_policy(pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4, title='Policy:'):
    print(title)
    arrs = {k:v for k,v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi(s)
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")


def print_state_value_function(V, P, n_cols=4, prec=3, title='State-value function:'):
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")


def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        while not done and steps < max_steps:
            state, _, done, h = env.step(pi(state))
            steps += 1
        results.append(state == goal_state)
    return np.sum(results)/len(results)


def mean_return(env, pi, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        results.append(0.0)
        while not done and steps < max_steps:
            state, reward, done, _ = env.step(pi(state))
            results[-1] += reward
            steps += 1
    return np.mean(results)


env = gym.make('FrozenLake-v1')
P = env.env.P
init_state = env.reset()
goal_state = 15
LEFT, DOWN, RIGHT, UP = range(4)

P


state, reward, done, info = env.step(RIGHT)
print("state:{0} - reward:{1} - done:{2} - info:{3}".format(state, reward, done, info))


pi_frozenlake = lambda s: {
    0: RIGHT,
    1: DOWN,
    2: RIGHT,
    3: LEFT,
    4: DOWN,
    5: LEFT,
    6: RIGHT,
    7:LEFT,
    8: UP,
    9: DOWN,
    10:LEFT,
    11:DOWN,
    12:RIGHT,
    13:RIGHT,
    14:DOWN,
    15:LEFT #Stop
}[s]
print_policy(pi_frozenlake, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)


print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_frozenlake, goal_state=goal_state) * 100,
    mean_return(env, pi_frozenlake)))


# Create your own policy (example: a simple one always going RIGHT unless near bottom row)
pi_2 = lambda s: {
    0: RIGHT,
    1: RIGHT,
    2: LEFT,
    3: DOWN,
    4: RIGHT,
    5: LEFT,
    6: RIGHT,
    7: DOWN,
    8: RIGHT,
    9: LEFT,
    10: RIGHT,
    11: DOWN,
    12: LEFT,
    13: DOWN,
    14: RIGHT,
    15: LEFT  # Terminal
}[s]

print("Name: mukesh r")
print("Register Number: 212223240100")
print_policy(pi_2, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)
print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_2, goal_state=goal_state) * 100,
    mean_return(env, pi_2)))


# Find the probability of success and the mean return of your policy

success_rate = probability_success(env, pi_2, goal_state=goal_state) * 100
avg_return = mean_return(env, pi_2)

print("Reaches goal {:.2f}% of the time.".format(success_rate))
print("Obtains an average undiscounted return of {:.4f}.".format(avg_return))


# Compare your policy with the first policy

# Evaluate first policy
success_frozenlake = probability_success(env, pi_frozenlake, goal_state=goal_state) * 100
avg_return_frozenlake = mean_return(env, pi_frozenlake)

# Evaluate your policy
success_pi2 = probability_success(env, pi_2, goal_state=goal_state) * 100
avg_return_pi2 = mean_return(env, pi_2)

print("----- Policy Comparison -----")
print("First Policy:")
print("  Success Rate: {:.2f}%".format(success_frozenlake))
print("  Mean Return : {:.4f}".format(avg_return_frozenlake))

print("\nYour Policy:")
print("  Success Rate: {:.2f}%".format(success_pi2))
print("  Mean Return : {:.4f}".format(avg_return_pi2))


def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        delta = 0
        for s in range(len(P)):
            v = 0
            a = pi(s)  # action chosen by policy
            for prob, next_state, reward, done in P[s][a]:
                v += prob * (reward + gamma * V[next_state] * (not done))
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V



# Code to evaluate the first policy
V1 = policy_evaluation(pi_frozenlake, P,gamma=0.99)
print_state_value_function(V1, P, n_cols=4, prec=5)


# Code to evaluate the second policy
V2 = policy_evaluation(pi_2, P, gamma=0.99)
print_state_value_function(V2, P, n_cols=4, prec=5)


# Comparing the two policies

# Evaluate both policies
V1 = policy_evaluation(pi_frozenlake, P, gamma=0.99)
V2 = policy_evaluation(pi_2, P, gamma=0.99)

# Compare state by state
better_or_equal = np.all(V2 >= V1)

print("Value Function of First Policy (pi_frozenlake):")
print_state_value_function(V1, P, n_cols=4, prec=5)

print("\nValue Function of Second Policy (pi_2):")
print_state_value_function(V2, P, n_cols=4, prec=5)

print("\nPolicy Comparison Result:")
if better_or_equal:
    print("pi_2 is better than or equal to pi_frozenlake (π₂ ≥ π₁)")
else:
    print("pi_frozenlake is better than or equal to pi_2 (π₁ ≥ π₂)")



V1>V2


if(np.sum(V1>V2)==11):
  print("The first policy is the better policy")
elif(np.sum(V2>V1)==11):
  print("The second policy is the better policy")
else:
  print("Both policies have their merits.")





```

## OUTPUT:

# Policy 1:
<img width="676" height="485" alt="image" src="https://github.com/user-attachments/assets/95197ebe-45eb-4034-8406-be9b82bf7f2d" />

# Policy 2:
<img width="834" height="196" alt="image" src="https://github.com/user-attachments/assets/a71de1ed-48b5-4b66-935c-5dcb5eb8f21b" />

# Policy Evaluation of Policies
<img width="502" height="426" alt="image" src="https://github.com/user-attachments/assets/45541127-d9ec-44f0-a934-99d250c38294" />

# Comparing Policies:
<img width="621" height="278" alt="image" src="https://github.com/user-attachments/assets/73135f1d-fb01-452e-ae5b-0e0d789cbfe0" />



## RESULT:

Therefore, policies are compared successfully using policy evaluation function in Frozen-Lake MDP.

