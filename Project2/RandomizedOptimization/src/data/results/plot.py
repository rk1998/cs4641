import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

def get_csv_data(filename, index=0):
    '''
    Gets data from a csv file and puts it into a pandas dataframe
    '''
    if os.path.exists(filename):
        print( filename + " found ")
        data_frame = pd.read_csv(filename, index_col=index)
        return data_frame
    else:
        print("file not found")


def plot_error_results(dataframe, algoName):
    data_set_sizes = dataframe["Dataset Size"].values
    training_error = dataframe["Training Error"].values
    testing_error = dataframe["Testing Error"].values
    plt.figure()
    plt.grid()
    plt.title("Neural Network trained with " + algoName + " Learning Curve")
    plt.plot(data_set_sizes, training_error, label='training_error',
        color='red')
    plt.plot(data_set_sizes, testing_error, label='testing_error', color='green')
    plt.legend()
    plt.xlabel("Dataset Sizes")
    plt.ylabel("Error")
    plt.show()

def compare_error_results(rhc, sa, ga, backprop):
    data_set_sizes = rhc["Dataset Size"].values
    training_error_rhc = rhc["Training Error"].values
    testing_error_rhc = rhc["Testing Error"].values
    training_error_sa = sa["Training Error"].values
    testing_error_sa = sa["Testing Error"].values
    training_error_ga = ga["Training Error"].values
    testing_error_ga = ga["Testing Error"].values
    training_error_backprop = backprop["Training Error"].values
    testing_error_backprop = backprop["Testing Error"].values
    plt.figure()
    plt.grid()
    plt.title("Training and Testing Error Comparison")
    plt.plot(data_set_sizes, training_error_rhc, label='trainRHC', color='green')
    plt.plot(data_set_sizes, testing_error_rhc, label='testRHC', color='red')
    plt.plot(data_set_sizes, training_error_sa, label='trainSA', color='orange')
    plt.plot(data_set_sizes, testing_error_sa, label='testSA', color='navy')
    plt.plot(data_set_sizes, training_error_ga, label='trainGA', color='black')
    plt.plot(data_set_sizes, testing_error_ga, label='testGA', color='yellow')
    plt.plot(data_set_sizes, training_error_backprop, label='trainBackProp', color='cyan')
    plt.plot(data_set_sizes, testing_error_backprop, label='testBackProp', color='magenta')
    plt.legend()
    plt.xlabel("Dataset Sizes")
    plt.ylabel("Error")
    plt.show()

def plot_optimization(dataframe, problemName):
    rhc = dataframe["RHC"].values
    sa = dataframe["SA"].values
    ga = dataframe["GA"].values
    mimic = dataframe["MIMIC"].values
    iterations = range(1, 51)
    plt.figure()
    plt.grid()
    plt.title(problemName + " Results")
    plt.plot(iterations, rhc, label='RHC', color='red')
    plt.plot(iterations, sa, label='SA', color='green')
    plt.plot(iterations, ga, label='GA', color='navy')
    plt.plot(iterations, mimic, label='MIMIC', color='darkorange')
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.show()

def compare_training_times(df1, df2, df3, df4, algo1, algo2, algo3, algo4):
    data_set_sizes = df1["Dataset Size"].values
    training_time1 = df1["Training Time"].values
    training_time2 = df2["Training Time"].values
    training_time3 = df3["Training Time"].values
    training_time4 = df4["Training Time"].values
    plt.figure()
    plt.grid()
    plt.title("Training Times")
    plt.plot(data_set_sizes, training_time1, label=algo1, color='navy')
    plt.plot(data_set_sizes, training_time2, label=algo2, color='darkorange')
    plt.plot(data_set_sizes, training_time3, label=algo3, color='green')
    plt.plot(data_set_sizes, training_time4, label=algo4, color='red')
    plt.legend()
    plt.xlabel("Dataset Sizes")
    plt.ylabel("Time")
    plt.show()

if sys.argv[1] == 'rhc':
    df1 = get_csv_data("rhcresults.csv", index=None)
    plot_error_results(df1, "Randomized Hill Climbing")
if sys.argv[1] == 'backprop':
    df1 = get_csv_data("backpropresults.csv", index=None)
    plot_error_results(df1, "Back Propagation")
if sys.argv[1] == 'rhcrestart':
    df1 = get_csv_data("rhcrestartresults.csv", index=None)
    plot_error_results(df1, "Random Restarts")
if sys.argv[1] == 'simanneal':
    df1 = get_csv_data("simannealresults.csv", index=None)
    plot_error_results(df1, "Simmulated Annealing")
if sys.argv[1] == 'simanneal2':
    df1 = get_csv_data("simannealresults2.csv", index=None)
    plot_error_results(df1, "Simmulated Annealing")
if sys.argv[1] == 'ga':
    df1 = get_csv_data("genalgresults.csv", index=None)
    plot_error_results(df1, "Genetic Algorithm")
if sys.argv[1] == 'fourpeaks':
    df1 = get_csv_data("fourpeaksresults.csv", index=None)
    plot_optimization(df1, "Four Peaks")
if sys.argv[1] == 'flipflop':
    df1 = get_csv_data("flipflopresults.csv", index=None)
    plot_optimization(df1, "Flip Flop")
if sys.argv[1] == 'travelingsalesman':
    df1 = get_csv_data("travelingsalesmanresults.csv", index=None)
    plot_optimization(df1, "Traveling Salesman")

if sys.argv[1] == 'times':
    df1 = get_csv_data("rhcresults.csv", index=None)
    df2 = get_csv_data("simannealresults.csv", index=None)
    df3 = get_csv_data("backpropresults.csv", index=None)
    df4 = get_csv_data("genalgresults.csv", index=None)
    compare_training_times(df1, df2, df3, df4, "RHC", "SA", "BackProp", "GA")

if sys.argv[1] == 'compare':
    df1 = get_csv_data("rhcresults.csv", index=None)
    df2 = get_csv_data("simannealresults.csv", index=None)
    df3 = get_csv_data("genalgresults.csv", index=None)
    df4 = get_csv_data("backpropresults.csv", index=None)
    compare_error_results(df1, df2, df3, df4)









