import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = 'Arial'

class Plotter:
    def __init__(self, test_name):
        self.fname = 'results/'+ test_name + '.csv'
        self.test_name = test_name
        self.df = pd.read_csv(self.fname)

        # plt setting
        self.xlabel_fontsize = 40
        self.ylabel_fontsize = 40
        self.xticks_fontsize = 38
        self.yticks_fontsize = 38
        self.legend_fontsize = 24
        self.lw = 3.0
        self.markersize = 16
        self.mew = 2.5

        self.markers = ['*', '>', 'x', '.', '^', '<', 'v']


    def patch_size(self):
        patch_size_list = [8,16, 32, 64]
        snr_range = np.arange(-20,21,2)

        plt.figure(figsize=(15, 12))

        for i, patch_size in enumerate(patch_size_list):
            plt.plot(snr_range, self.df[str(patch_size)].to_list(), lw=self.lw, label=f'patch_size 2x{str(patch_size)}', marker=self.markers[i], markersize=self.markersize)

        plt.xlabel("Signal to Noise Ratio", fontsize=self.xlabel_fontsize)
        plt.ylabel("Classification Accuracy", fontsize=self.ylabel_fontsize)
        plt.xticks(fontsize=self.xticks_fontsize)
        plt.yticks(fontsize=self.yticks_fontsize)
        plt.legend(loc='upper left', framealpha=1, fontsize=self.legend_fontsize)
        
        plt.ylim(0.0, 1.0)

        plt.grid()
        plt.savefig(f'../figures/{self.test_name}.png', bbox_inches='tight')
    
    def sample_size(self):
        # Read csv
        columns = [16, 32, 64, 128, 256, 512, 1024]

        snr_range = range(-20, 21, 2)
        results = [self.df[str(i)].to_list() for i in columns]

        plt.figure(figsize=(15, 12))

        for i, model in enumerate(columns):
            plt.plot(snr_range, results[i], label=f'sample_size {model}', marker=self.markers[i],
                    markersize=self.markersize, mew=self.mew)

        plt.xlabel("Signal to Noise Ratio", fontsize=self.xlabel_fontsize)
        plt.ylabel("Classification Accuracy", fontsize=self.ylabel_fontsize)
        plt.xticks(fontsize=self.xticks_fontsize)
        plt.yticks(fontsize=self.yticks_fontsize)
        plt.legend(loc='upper left', framealpha=1, fontsize=self.legend_fontsize)

        plt.ylim(0.0, 1.0)

        plt.grid()

        plt.savefig(f'../figures/{self.test_name}.png', bbox_inches='tight')
    

    def sample_size_compare(self):
        # Read csv
        columns = [64, 128, 256, 512, 1024]
        cnn_df = pd.read_csv('results/cnn_input_size.csv')
        res_df = pd.read_csv('results/resnet_input_size.csv')

        snr_range = range(-20, 21, 2)
        results = [self.df[str(i)].to_list() for i in columns]
        cnn_results = [cnn_df[str(i)].to_list() for i in columns]
        res_results = [res_df[str(i)].to_list() for i in columns]

        plt.figure(figsize=(15, 12))

        # for i, model in enumerate(columns):
        #     plt.plot(snr_range, results[i], 'C0',label=f'Proposed_sample_size {model}', marker=self.markers[i],
        #             markersize=self.markersize, mew=self.mew, linestyle='solid')

        for i, model in enumerate(columns):
            plt.plot(snr_range, res_results[i], 'C1',label=f'ResNet_sample_size {model}', marker=self.markers[i],
                    markersize=self.markersize, mew=self.mew, linestyle='dotted')
        
        for i, model in enumerate(columns):
            plt.plot(snr_range, cnn_results[i], 'C2', label=f'CNN_sample_size {model}', marker=self.markers[i],
                    markersize=self.markersize, mew=self.mew, linestyle='dashed')
            

        plt.xlabel("Signal to Noise Ratio", fontsize=self.xlabel_fontsize)
        plt.ylabel("Classification Accuracy", fontsize=self.ylabel_fontsize)
        plt.xticks(fontsize=self.xticks_fontsize)
        plt.yticks(fontsize=self.yticks_fontsize)
        plt.legend(loc='upper left', framealpha=1, fontsize=22)

        plt.grid()

        plt.savefig(f'../figures/size_compare_1.png', bbox_inches='tight')
    
    def model_compare(self):
        model_list = self.df.columns
        snr_range = np.arange(-20,21,2)

        plt.figure(figsize=(15, 12))

        for i, model in enumerate(model_list):
            plt.plot(snr_range, self.df[model].to_list(), lw=self.lw, label=f'{model}', marker=self.markers[i], markersize=self.markersize)

        plt.xlabel("Signal to Noise Ratio", fontsize=self.xlabel_fontsize)
        plt.ylabel("Classification Accuracy", fontsize=self.ylabel_fontsize)
        plt.xticks(fontsize=self.xticks_fontsize)
        plt.yticks(fontsize=self.yticks_fontsize)
        plt.legend(loc='upper left', framealpha=1, fontsize=self.legend_fontsize)
        
        plt.grid()
        plt.savefig(f'../figures/{self.test_name}.png', bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('test', type=str, default='512_patch', help='Enter csv file name')

    args = parser.parse_args()
    plotter = Plotter(args.test)
    #plotter.model_compare()
    #plotter.sample_size()
    plotter.sample_size_compare()
    #plotter.patch_size()

