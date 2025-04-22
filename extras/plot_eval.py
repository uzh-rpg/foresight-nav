"""
Generate plots for evaluation of Imagination layer.

Usage:
python eval_plots.py \
    --file='Evaluation_Logs/results_ test_unet_BCE_run2.txt' \
    --save_dir='Evaluation_Logs/plots/'
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def get_times(file_path):
    """
    Get the times from the file.
    Format: scene_00325, Vanilla: 1000, Imagine: 1000
    """
    vanilla_times = []
    imagine_times = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            vanilla_times.append(int(line[1].split(':')[1].strip()))
            imagine_times.append(int(line[2].split(':')[1].strip()))
    
    return vanilla_times, imagine_times

def plot_times_as_hist(vanilla_times, imagine_times, vanilla_completion_rate, imagine_completion_rate, plt_title):
    """
    Plot the times as histogram.
    """
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(10, 5))

    bins = np.linspace(0, 3000, 25)
    sns.histplot(vanilla_times, bins=bins, color='blue', label='Vanilla', ax=ax)
    sns.histplot(imagine_times, bins=bins, color='red', label='Imagine', ax=ax)

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Time taken for Vanilla and Imagine: {plt_title}')

    # add mean and std and completion rate
    # ax.text(0.5, 0.9, f'Vanilla Mean: {np.mean(vanilla_times):.2f}', transform=ax.transAxes)
    # ax.text(0.5, 0.85, f'Imagine Mean: {np.mean(imagine_times):.2f}', transform=ax.transAxes)
    # ax.text(0.5, 0.8, f'Vanilla Std: {np.std(vanilla_times):.2f}', transform=ax.transAxes)
    # ax.text(0.5, 0.75, f'Imagine Std: {np.std(imagine_times):.2f}', transform=ax.transAxes)
    # ax.text(0.5, 0.7, f'Vanilla Completion Rate: {vanilla_completion_rate:.2f}', transform=ax.transAxes)
    # ax.text(0.5, 0.65, f'Imagine Completion Rate: {imagine_completion_rate:.2f}', transform=ax.transAxes)

    ax.legend()
    return fig

def main(args):
    vanilla_times, imagine_times = get_times(args.file)
    vanilla_times = np.array(vanilla_times)
    imagine_times = np.array(imagine_times)

    # get completion rate for vanilla and imagine (3000 time limit)
    vanilla_completion_rate = np.sum(vanilla_times < 3000) / len(vanilla_times)
    imagine_completion_rate = np.sum(imagine_times < 3000) / len(imagine_times)
    print('Vanilla Completion Rate:', vanilla_completion_rate)
    print('Imagine Completion Rate:', imagine_completion_rate)

    # reduce the times to only those that completed
    vanilla_times = vanilla_times[vanilla_times < 3000]
    imagine_times = imagine_times[imagine_times < 3000]
    print(len(vanilla_times))
    print(len(imagine_times))
    # print the mean and std
    print('Vanilla Mean:', np.mean(vanilla_times))
    print('Imagine Mean:', np.mean(imagine_times))
    print('Vanilla Std:', np.std(vanilla_times))
    print('Imagine Std:', np.std(imagine_times))

    # plot title
    plt_title = args.file.split('/')[-1].split('.')[0].replace('results_test_', '')

    # plot the times as histogram
    fig = plot_times_as_hist(vanilla_times, imagine_times, vanilla_completion_rate, imagine_completion_rate, plt_title)
    
    # save the figure
    plot_name = args.file.split('/')[-1].split('.')[0]
    fig.savefig(os.path.join(args.save_dir, f'{plot_name}.png'))

def config():
    parser = argparse.ArgumentParser(description='Plot evaluation results')
    parser.add_argument('--file', type=str, help='File path to evaluation results')
    parser.add_argument('--save_dir', type=str, help='Directory path to save plot')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = config()
    main(args)
