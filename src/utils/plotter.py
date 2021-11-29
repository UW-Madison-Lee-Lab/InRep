""" Plotting figures
	@date 08/14/2020
"""

import os
import pandas
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



class Plotter:
	small_font, medium_font, big_font, figure_size, figure_style = None, None, None, None, None
	fig_path = None
	color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
	tab_color = ["tab:blue", "tab:green", "tab:red", "tab:cyan", "tab:pink", "tab:olive", "tab:gray"]

	def __init__(self, fig_path, small=8.5, medium=14, big=20, fig_size=(7, 7), style="ggplot"):
		"""
		Constructor for Plotter instance

		:param fig_path: path to store figures (Directory should exist)
		:param small: size of small font
		:param medium: size of medium font
		:param big: size of big font
		:param fig_size: size of figure
		:param style: plot style
		"""
		# Save setups
		self.small_font, self.medium_font, self.big_font = small, medium, big
		self.figure_size, self.figure_style = fig_size, style

		# Check for directory
		if os.path.isdir(fig_path):
			self.fig_path = os.path.abspath(fig_path)
		else:
			raise SystemExit("Check Figure Directory and Retry. Directory Not Exist")

	def __plot_setting(self):
		""" Private helper method to setup new figure

		:return: new plt.figure
		"""
		plt.style.use(self.figure_style)
		plt.tight_layout()
		plt.rc('figure', titlesize=self.big_font)  # fontsize of the figure title
		# plt.rc('font', size=self.medium_font)  # controls default text sizes
		# plt.rc('axes', titlesize=self.medium_font)  # fontsize of the axes title
		# plt.rc('axes', labelsize=self.medium_font)  # fontsize of the x and y labels
		# plt.rc('xtick', labelsize=self.small_font)  # fontsize of the tick labels
		# plt.rc('ytick', labelsize=self.small_font)  # fontsize of the tick labels
		# plt.rc('legend', fontsize=self.medium_font)  # legend fontsize
		# plt.rc('text', usetex=True)
		return plt.figure(figsize=self.figure_size)

	def plot_train_loss_curves(self, loss_csv, filename, title="Train Loss"):
		""" Method that draws loss curve

		:param loss_csv: csv file contains training time loss
		:param title: title of the figure (Default: Train Loss)
		:param filename: name of plot to be saved
		"""
		# Load CSV data
		if os.path.isfile(loss_csv):
			data = pandas.read_csv(loss_csv)
		else:
			raise SystemExit("%s file not found.".format(loss_csv))

		filepath = os.path.join(self.fig_path, filename)  # get figure save path
		loss_name = data.columns.drop("epoch")  # Retrieve list of loss

		# Draw Figure
		fig = self.__plot_setting()
		ax = fig.add_subplot(1, 1, 1)
		for i in range(len(loss_name)):
			ax.plot(data[["epoch"]], data[[loss_name[i]]], label=loss_name[i],
					c=self.color[i % len(self.color)])
		plt.title(title)
		plt.xlabel("epoch")
		plt.ylabel("loss")

		# Legend
		plt.legend(loc='best')

		# Save Figure
		plt.savefig(os.path.join(filepath))
		plt.close(fig)

	@staticmethod
	def __raw_result_to_plot_data(raw_df, class_index=-1):
		""" Read raw experiment result and convert it to plot-ready dataframe
		raw result reports (x,y) pairs with duplicated x
		plot-ready dataframe should contains x and multiple y values in each row

		:param raw_df: raw experiment result
		:return plot_df: plot-ready dataframe
		"""
		if class_index == -1:
			class_index = 0
		# retrieve unique x
		x = raw_df.iloc[:, 0].unique()
		df_source = {}  # temporal place to re-structure raw data

		# Iterate through all datapoint and categorize by x
		for current_x in x:
			y_index = raw_df.index[raw_df.iloc[:, 0] == current_x].tolist()
			df_source.update({current_x: raw_df.loc[y_index].iloc[:, class_index+1].tolist()})

		return pandas.DataFrame.from_dict(df_source)

	def plot_exp_acgan_reprogram_compare(self, log_acgan, log_reprogram, log_real, filename, xlab, ylab,
										 title="AC-GAN vs Reprogram", class_index=-1):
		""" plotting figure to compare AC-GAN and Reprogram score

		:param log_acgan: path of log file contains score of AC-GAN
		:param log_reprogram: path of log file contains score of Reprogram
		:param filename: name of plot to be saved
		:param xlab: x label description
		:param ylab: y label description
		:param title: title of the figure (Default: "AC-GAN vs Reprogram")
		"""
		# Load logs
		if os.path.isfile(log_acgan):
			acgan_raw = pandas.read_csv(log_acgan, delimiter=" ", header=None, engine='python')
		else:
			raise SystemExit("{} file not found.".format(log_acgan))
		if os.path.isfile(log_reprogram):
			reprogram_raw = pandas.read_csv(log_reprogram, delimiter=" ", header=None, engine='python')
		else:
			raise SystemExit("{} file not found.".format(log_reprogram))
		if os.path.isfile(log_real):
			real_raw = pandas.read_csv(log_real, delimiter=" ", header=None, engine='python')
		else:
			raise SystemExit("{} file not found.".format(log_real))


		# Data re-structuring
		acgan_data = self.__raw_result_to_plot_data(acgan_raw, class_index)
		reprogram_data = self.__raw_result_to_plot_data(reprogram_raw, class_index)
		real_data = self.__raw_result_to_plot_data(real_raw, class_index)

		# Draw Figure
		fig = self.__plot_setting()
		ax = fig.add_subplot(1, 1, 1)
		# ax.set_xscale('log')
		# ax.set_yscale('log')



		# Lines
		ax.plot(acgan_data.columns, acgan_data.median(), label="AC-GAN", c=self.color[1])
		ax.plot(reprogram_data.columns, reprogram_data.median(), label="Reprogram", c=self.color[2])
		ax.plot(real_data.columns, real_data.median(), label="Real", c=self.color[3])

		# Box plot
		bp_acgan = ax.boxplot(acgan_data.to_numpy(), positions=acgan_data.columns.tolist(),
							  patch_artist=True, widths=np.repeat(2,  len(acgan_data.columns.tolist())),
							  flierprops=dict(markeredgecolor=self.color[1], markersize=4))
		bp_reprogram = ax.boxplot(reprogram_data.to_numpy(), positions=reprogram_data.columns.tolist(),
								  patch_artist=True, widths=np.repeat(2, len(acgan_data.columns.tolist())),
								  flierprops=dict(markeredgecolor=self.color[2], markersize=4))
		bp_real = ax.boxplot(real_data.to_numpy(), positions=real_data.columns.tolist(),
								  patch_artist=True, widths=np.repeat(2, len(acgan_data.columns.tolist())),
								  flierprops=dict(markeredgecolor=self.color[2], markersize=4))
		for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
			plt.setp(bp_acgan[element], color=self.color[1])
			plt.setp(bp_reprogram[element], color=self.color[2])
			plt.setp(bp_real[element], color=self.color[3])

		for patch in bp_acgan['boxes']:
			patch.set_facecolor(self.tab_color[1])

		for patch in bp_reprogram['boxes']:
			patch.set_facecolor(self.tab_color[2])

		for patch in bp_real['boxes']:
			patch.set_facecolor(self.tab_color[3])

		# Title and labels
		filepath = os.path.join(self.fig_path, filename)  # get figure save path
		plt.title(title)
		plt.xlabel(xlab)
		plt.ylabel(ylab)

		# Legend
		plt.legend(loc='best')

		# Save Figure
		plt.savefig(os.path.join(filepath), bbox_inches='tight', pad_inches=0.1)
		plt.close(fig)


def plot(args):
	"""
	plot experimental results (Use plot_exp_acgan_reprogram_compare() method)
	"""
	# exp_mode and eval_mode dictionary
	exp_dictionary = {0: "Complexity", 1: "Symmetric-Noise", 2: "Asymmetric-Noise"}
	eval_dictionary = {0: "FID", 2: "FittingCapacity"}
	xlab_dictionary = {0: "number of labeled sample", 1: "noise ration (value = x/50)",
					   2: "noise ration (value = x/50)"}
	# Argument for plotting function
	eval_path = "../../results/reports/evals/"
	# Generate plot
	plot_size = (args.width, args.height)
	plotter = Plotter("../../results/reports/plots", fig_size=plot_size)
	plot_log_acgan = eval_path + "scores_gan2-exp" + str(args.exp_mode) + "-{}-n2-eval".format(args.dataset) + \
					 str(args.eval_mode) + ".txt"
	plot_log_reprogram = eval_path + "scores_gan1-exp" + str(args.exp_mode) + "-{}-n2-eval".format(args.dataset) + \
						 str(args.eval_mode) + ".txt"
	plot_log_real = eval_path + "scores_gan3-exp" + str(args.exp_mode) + "-{}-n2-eval".format(args.dataset) + \
						 str(args.eval_mode) + ".txt"
	plot_xlab = xlab_dictionary.get(args.exp_mode)
	plot_ylab = eval_dictionary.get(args.eval_mode)
	plot_title = exp_dictionary.get(args.exp_mode) + " (" + eval_dictionary.get(args.eval_mode) + ")"
	class_index=-1
	if args.eval_mode == 0:
		for c in range(2):
			class_index = c
			plot_title = plot_title + " (Class: {})".format(class_index)
			plot_filename = exp_dictionary.get(args.exp_mode) + "_" + eval_dictionary.get(args.eval_mode) + "_c{}.png".format(class_index)
			plotter.plot_exp_acgan_reprogram_compare(plot_log_acgan, plot_log_reprogram, plot_log_real, plot_filename, plot_xlab, plot_ylab, plot_title, class_index)
	else:
		 plot_filename = exp_dictionary.get(args.exp_mode) + "_" + eval_dictionary.get(args.eval_mode) + ".png"
		 plotter.plot_exp_acgan_reprogram_compare(plot_log_acgan, plot_log_reprogram, plot_log_real, plot_filename, plot_xlab, plot_ylab, plot_title, -1)

def plot_gaussian(
	ax,
	data,
	codes=None,
	color='w',
	size=5,
	color_palette=sns.color_palette('Set1', n_colors=8,  desat=.5)):
	sns.set_style('white')
	sns.set(color_codes=True)

	ax.set_aspect('equal')
	ax.set_ylim((-size, size))
	ax.set_xlim((-size, size))
	ax.tick_params(labelsize=10)
	if codes is not None:
		c = [color_palette[i] for i in codes]
		color = None
	else:
		c = None
	sns.kdeplot(data[:, 0], data[:, 1],
				 cmap='Blues', shade=True, shade_lowest=False, ax=ax)
	ax.scatter(data[:, 0], data[:, 1], linewidth=1, marker='+', c=c, color=color)

def plot_toy(gm_data, gm_labels, save_path):
	fig, ax = plt.subplots(figsize=(6, 6))
	ax.set_title('Mixture of Gaussians Dataset')
	plot_gaussian(ax, gm_data, codes=gm_labels)
	plt.savefig(save_path)
	plt.close()

	## https://apps.automeris.io/wpd/ to extract data from figures

def preprocess_raw_txt(path_rc = '/content/Ours.txt', 
                       path_ac = '/content/ACGANs.txt', 
                       path_proj = '/content/ProjGANs.txt'):
  '''
  df_list contains the data from all paths

  read txt file of the form 
  0 20.4
  10 40.5

  '   '
  '   '
  '   '

  '''
  df_list = []
  rc_raw = pandas.read_csv(path_rc, sep=" ", header=None, names=["nsamples", "score"])
  ac_raw = pandas.read_csv(path_ac, sep=" ", header=None, names=["nsamples", "score"])
  projraw = pandas.read_csv(path_proj, sep=" ", header=None, names=["nsamples", "score"])

  for raw in [rc_raw, ac_raw, projraw]:
    xlabels = raw.iloc[:,0].unique()
    df_source = {}  # temporal place to re-structure raw data

    # Iterate through all datapoint and categorize by x
    for xlabel in xlabels:
      y_index = raw.index[raw.iloc[:, 0] == xlabel].tolist()
      df_source.update({xlabel: raw.loc[y_index].iloc[:,1].tolist()})

    df_list.append(pandas.DataFrame.from_dict(df_source))

  return df_list, xlabels


def plot_figure(save_path = 'fig.pdf',
		path_rc = '/content/Ours.txt', 
                path_ac = '/content/ACGANs.txt', 
                path_proj = '/content/ProjGANs.txt',
                plot_sigma = True,
                plot_smooth = False,
                alpha=0.15):
  
  '''
  Possible pacakages for the function to work:

  !sudo apt install texlive-fonts-recommended texlive-fonts-extra
  !sudo apt install dvipng
  !sudo apt install texlive-full
  !sudo apt install texmaker
  !sudo apt install ghostscript

  '''
  
#   matplotlib inline
  import matplotlib
  import matplotlib.pyplot as plt
  from scipy.interpolate import make_interp_spline
  import matplotlib.font_manager as font_manager
  import numpy as np
  import pandas

  plt.rc('font', family='serif', serif='Times')
  plt.rc('font', variant='small-caps')
  plt.rc('text', usetex=True)
  plt.rc('xtick', labelsize=10)
  plt.rc('ytick', labelsize=10)
  plt.rc('axes', labelsize=10)

  full_names = ['RepGANs', 'ACGANs', 'ProjGANs']
  c = ['darkblue', 'darkgreen', 'darkorange']
  m = ['s', 'o', '^']

  df_list, xlabels = preprocess_raw_txt(path_rc = '/content/Ours.txt', 
                                        path_ac = '/content/ACGANs.txt', 
                                        path_proj = '/content/ProjGANs.txt')

  x = np.arange(len(xlabels))  # the label locations
  xlabels = [labels.replace('%', '$\%$') for labels in xlabels] #display % sign

  '''
  change x here to change the location of x ticks
  '''

  fig, ax = plt.subplots()
  for i in range(len(df_list)):
      data_array = df_list[i].to_numpy()
      '''
      data_array is of the form:

      [
        [*, *, *, *, *] <-- first run of each score
        [] <-- second run (if any)
        [] <-- third run (if any)
      ]
      '''
      y = data_array[0] #can change to max, mean, along axis=0

      if plot_smooth:
        x_smooth = np.linspace(x.min(), x.max(), 100) 
        spl = make_interp_spline(x, y, k=2)
        y_smooth = spl(x_smooth)
        markon = [0, 20, 40, 60, 80, 99]
        ax.plot(x_smooth, y_smooth, label=full_names[i], color=c[i], marker=m[i], markevery=markon)
      else:
        ax.plot(x, y, marker=m[i], label=full_names[i], color=c[i])

      if plot_sigma:
        mu = np.mean(data_array, axis=0)
        sigma = np.std(data_array, axis=0)
        ax.fill_between(x, mu+sigma, mu-sigma, facecolor=c[i], alpha=alpha)

  # Add some text for labels, title and custom x-axis tick labels, etc.
  ax.set_xlabel('Number of labeled samples',size=12)
  ax.set_ylabel('Scores ($\%$)',size=12)
  ax.set_ylim([0,105])
  #ax.set_xlim([0,100])
  ax.set_xticks(x)
  ax.set_xticklabels(xlabels)

  font = font_manager.FontProperties(variant='small-caps')
  
  handles, labels = ax.get_legend_handles_labels()
  fig.legend(handles, labels, bbox_to_anchor=(.05, 1.26), ncol=3, loc='upper left', prop=font)
  ax.grid(True)

  fig.tight_layout()
  width = 3.487
  height = width / 1.618

  fig.set_size_inches(width, height)
  fig.savefig(save_path, bbox_inches='tight')

def offset(x, inds):
	for i in range(len(inds)):
		if x == inds[i]:
			return i

def process_intra_fid(filepath, inds):
	scores = np.zeros((len(inds), 6))
	with open(filepath) as fp:
		for line in fp:
			if ',' in line:
				x = line.strip('\n')
				x = x.replace('[', '')
				x = x.replace(']', '')
				x = x.replace(',', '')
				arr = x.split(' ')
				num = float(arr[0])
				
				values = np.asarray([float(e) for e in arr[1:]])
				idx = offset(num, inds)
				scores[idx, :] = [num, np.mean(values), np.std(values), np.median(values), np.min(values), np.max(values)]
	# print(scores)
	return scores 

if __name__ == '__main__':
	# Parse arguments
	# parser = argparse.ArgumentParser(description="Argument for Plotter")
	# # Image Size
	# parser.add_argument('--width', type=float, default=10, help="Width of Image")
	# parser.add_argument('--height', type=float, default=5, help="Height of Image")
	# # Log file Directory
	# parser.add_argument('--exp_mode', type=int, default=-1, help="Experiment Mode Code")
	# parser.add_argument('--eval_mode', type=int, default=2, help="Evaluation Mode Code")
	# # Get Argument
	# args = parser.parse_args()
	# args.dataset = 'cifar10'


	# if args.exp_mode > -1:
	# 	plot(args)
	# else:
	# 	for exp in [0, 1]:
	# 		args.exp_mode = exp
	# 		for eval in [2]:
	# 			args.eval_mode = eval
	# 			print("Plot exp ", exp, " on ", eval )
	# 			plot(args)
	inds = [0.1, 1.0]
	for g in [1]:
		filepath = '../../results/evals/intra-fid/exp1_e6-d4_200-g{}.txt'.format(g)
		csv_path = '../../results/csv/exp1_e6-d4_200-g{}.csv'.format(g)
		scores = process_intra_fid(filepath, inds)
		pandas.DataFrame(scores).to_csv(csv_path, header=['fraction', 'mean', 'std', 'median', 'min', 'max'], float_format="%.3f")