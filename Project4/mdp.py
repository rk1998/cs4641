import gym
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd


def print_actions(action_table):
    print(str(action_table[0:4]))
    print(str(action_table[4:8]))
    print(str(action_table[8:12]))
    print(str(action_table[12:16]))

def change_frozen_lake_reward(flenv, reward_for_hazard= -1.0, reward_for_goal=5.0):
    '''
    Changes the reward for the frozen lake problem

    '''
    hazard_states = [5, 7, 11, 12]
    transition_model = flenv.env.P
    for state in range(0, flenv.env.nS):
        print("State: " + str(state))
        for action in range(0, flenv.env.nA):
            transition_model = flenv.env.P[state][action]
            for i in range(0, len(transition_model)):
                if transition_model[i][1] in hazard_states:
                    new_transition = (transition_model[i][0], transition_model[i][1], reward_for_hazard, transition_model[i][3])
                    transition_model[i] = new_transition
                elif transition_model[i][1] == 15:
                    new_transition = (transition_model[i][0], transition_model[i][1], reward_for_goal, transition_model[i][3])
                    transition_model[i] = new_transition
            print(flenv.env.P[state][action])
    return flenv

def compute_value(state, action, transition_model, value_table, gamma):
    '''
    Computes utility of a state, action pair using the bellman equation
    V(s) = Sum(T(s, a, s')(R(s, a, s') + gamma * V'(s')))
    '''
    transition_list = transition_model[state][action]
    sum = 0
    for transition in transition_list:
        prob, nextstate, reward, done = transition
        sum = sum + (prob * (reward + gamma * value_table[nextstate] * (not done)))
    return sum

def compute_value_for_state(state, actions_from_state, transition_model,
    value_table, gamma):
    '''
    Determines the best action, and Max utility for a state
    '''
    max_value = -9999999999999
    best_action = None
    for action in actions_from_state:
        value = compute_value(state, action, transition_model, value_table,
            gamma)
        if value > max_value:
            max_value = value
            best_action = action
    return best_action, value


def value_iteration(environment, value_table, gamma=0.9, iterations=100,
    theta=0.01):
    '''
    Runs value iteration on the environment
    '''
    transition_model = environment.env.P
    action_space = range(0, environment.env.nA)
    action_table = [None] * environment.env.nS
    for i in range(0, iterations):
        delta = 0
        #new_values = np.zeros(environment.env.nS)
        new_values = np.random.rand(environment.env.nS)
        #new_values = value_table
        for state in range(0, environment.env.nS):
            action_val_pair = compute_value_for_state(state, action_space,
                transition_model, value_table, gamma)
            new_values[state] = action_val_pair[1]
            action_table[state] = action_val_pair[0]
            diff = abs(value_table[state] - new_values[state])
            delta = max(delta, diff)
        if delta <= theta:
            break
        value_table = new_values
    return new_values, action_table, i

def create_policy(environment, value_table):
    '''
    Gives you the optimal policy for each state given a
    value table
    '''
    policy_table = {}
    for state in range(0, environment.env.nS):
        policy_table[state] = value_table[state][0]
    return policy_table


def calculate_policy(environment, policy, gamma, max_iter=1000, theta=1e-3):
    '''
    Calculates Value Table for Policy
    '''
    old_value_function = np.random.rand(environment.env.nS)
    new_value_function = np.zeros(environment.env.nS)
    transition_model = environment.env.P
    for i in range(0, max_iter):
        delta = 0
        for state in range(0, environment.env.nS):
            state_util = 0
            actions = [policy[state]]
            for action in actions:
                state_util += compute_value(state, action, transition_model,
                    old_value_function, gamma)
            diff = abs(old_value_function[state] - state_util)
            delta = max(delta, diff)
            new_value_function[state] = state_util
        if delta <= theta:
            break
        old_value_function = new_value_function
    return new_value_function, i

def policy_iteration(environment, gamma=0.9, iterations=100):
    '''
    Runs policy iteration on the environment to find the optimal policy
    '''
    policy = np.zeros(environment.env.nS, dtype='int')
    converged=False
    iterations_to_converge = 0
    value_iterations = 0
    value_table = np.zeros(environment.env.nS)
    while not converged:
        value_table, iterations = calculate_policy(environment, policy, gamma)
        value_iterations += iterations
        for state in range(0, environment.env.nS):
            previous_action = policy[state]
            max = -999999999
            best_action = None
            for action in range(environment.env.nA):
                value = 0
                for transition in environment.env.P[state][action]:
                    prob = transition[0]
                    reward = transition[2]
                    value += prob * (reward + gamma * value_table[transition[1]])
                if value > max:
                    max = value
                    best_action = action
            if best_action != previous_action:
                converged = False
            else :
                converged = True
            policy[state] = best_action
            iterations_to_converge += 1
    return policy, value_table, iterations_to_converge
def run_policy(environment, policy, iterations=1000, env_type='frozenlake'):
    '''
    Runs the Policy on the given environment
    '''
    successes = 0
    failures = 0
    current_reward = 0
    steps_to_goal = 0
    for i in range(0, iterations):
        cur_state = environment.reset()
        done = False
        reward = 0
        for t in range(0, iterations):
            next_state, reward, done, info = environment.step(policy[cur_state])
            cur_state = next_state
            #current_reward += reward
            if done:
                current_reward += reward
                break
        if env_type == 'frozenlake':
            if cur_state == 15:
                successes += 1
            else:
                failures += 1
        elif env_type == 'taxi':
            steps_to_goal += t
    if env_type == 'frozenlake':
        return float(current_reward)/500.0, float(successes)/500.0, float(failures)/500.0
    else:
        return float(current_reward)/500.0, float(steps_to_goal)/500.0

if sys.argv[1] == 'frozenlake-val':
    if sys.argv[2] == 'normal':
        num_rounds = 300
        env = gym.make('FrozenLake-v0')
        env.reset()
        env = change_frozen_lake_reward(env, reward_for_hazard = -0.5, reward_for_goal=2.5)
        total_results = []
        for round in range(0, num_rounds):
            #value_table = np.random.rand(env.env.nS)
            iterations = pow(10, round/(100 + 1))
            value_table = np.zeros(env.env.nS)
            policy = value_iteration(env, value_table, iterations=int(iterations), theta=1e-5,
                gamma=0.9)
            results = run_policy(env, policy[1], iterations=500)
            if round % 10 == 0:
                print("Converged in " + str(policy[2]) + " iterations")
                print(results)
            current_results = [round, results[0], results[1], policy[2]]
            total_results.append(current_results)

        total_results_frame = pd.DataFrame(total_results)
        total_results_frame.to_csv("frozenlake_val_normal0.9.csv", encoding='utf-8', index=False)

    elif sys.argv[2] == 'negative':
        env = gym.make('FrozenLake-v0')
        env.reset()
        env = change_frozen_lake_reward(env, reward_for_hazard = -1.0, reward_for_goal=10.0)
        num_rounds = 300
        total_results = []
        for round in range(0, num_rounds):
            #value_table = np.random.rand(env.env.nS)
            value_table = np.zeros(env.env.nS)
            iterations = pow(10, round/100 + 1)
            policy = value_iteration(env, value_table, iterations=int(iterations), theta=1e-5, gamma=0.1)
            results  = run_policy(env, policy[1], iterations=500)
            if round % 10 == 0:
                print("Converged in " + str(policy[2]) + " iterations")
                print(results)
                print_actions(policy[1])
            current_results = [round, results[0], results[1], policy[2]]
            total_results.append(current_results)
        total_results_frame = pd.DataFrame(total_results)
        total_results_frame.to_csv("frozenlake_val_negative0.1.csv", encoding='utf-8', index=False)
elif sys.argv[1] == 'frozenlake-poly':
    if sys.argv[2] == 'normal':
        env = gym.make('FrozenLake-v0')
        env.reset()
        env = change_frozen_lake_reward(env, reward_for_hazard=-0.5, reward_for_goal=2.5)
        num_rounds = 300
        total_results = []
        for round in range(0, num_rounds):
            iterations = pow(10, round/100 + 1)
            policy = policy_iteration(env, gamma=0.1, iterations=int(iterations))
            results = run_policy(env, policy[0], iterations=500)
            if round % 10 == 0:
                print("Converged in " + str(policy[2]) + " iterations")
                print(results)
            current_results = [round, results[0], results[1], policy[2]]
            total_results.append(current_results)
        total_results_frame = pd.DataFrame(total_results)
        total_results_frame.to_csv("frozenlake_poly_normal0.1.csv", encoding='utf-8', index=False)
    elif sys.argv[2] == 'negative':
        env = gym.make('FrozenLake-v0')
        env.reset()
        env = change_frozen_lake_reward(env, reward_for_hazard = -1.0, reward_for_goal=10.0)
        num_rounds = 300
        total_results = []
        for round in range(0, num_rounds):
            iterations = pow(10, round/100 + 1)
            policy = policy_iteration(env, gamma=0.1, iterations=int(iterations))
            results = run_policy(env, policy[0], iterations=500)
            if round % 10 == 0:
                print("Converged in " + str(policy[2]) + " iterations")
                print(results)
            current_results = [round, results[0], results[1], policy[2]]
            total_results.append(current_results)
        total_results_frame = pd.DataFrame(total_results)
        total_results_frame.to_csv("frozenlake_poly_negative0.1.csv", encoding='utf-8', index=False)
elif sys.argv[1] == 'taxi-val':
    env = gym.make('Taxi-v2')
    env.reset()
    num_rounds = 300
    total_results = []
    for round in range(0, num_rounds):
        iterations = pow(10, round/100 + 1)
        value_table = np.zeros(env.env.nS)
        policy = value_iteration(env, value_table, iterations=int(iterations), theta=1e-5,
        gamma=0.5)
        results = run_policy(env, policy[1], env_type='taxi')
        if round % 10 == 0:
            print("Converged in " + str(policy[2]) + " iterations")
            print(results)
        current_results = [round, results[0], results[1], policy[2]]
        total_results.append(current_results)
    total_results_frame = pd.DataFrame(total_results)
    total_results_frame.to_csv("taxi_val0.5.csv", encoding='utf-8', index=False)
elif sys.argv[1] == 'taxi-poly':
    env = gym.make('Taxi-v2')
    env.reset()
    num_rounds = 300
    total_results = []
    for round in range(0, num_rounds):
        iterations = pow(10, round/100 + 1)
        policy = policy_iteration(env, gamma=0.9, iterations=int(iterations))
        results = run_policy(env, policy[0], env_type='taxi')
        if round % 10 == 0:
            print("Converged in " + str(policy[2]) + " iterations")
            print(results)
        current_results = [round, results[0], results[1], policy[2]]
        total_results.append(current_results)
    total_results_frame = pd.DataFrame(total_results)
    total_results_frame.to_csv("taxi_poly0.9.csv", encoding='utf-8', index=False)