import math
import warnings
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
import graphviz
from .dt_algo import Algo
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.offline as pyo
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import logging


class PlotNode:
    def __init__(self, plot, index, parent, even):
        """
        Instance constructor.

        @param plot: A boolean value to represent whether to create a plot at the current node.
        @param index: A pandas series to represent the index of the dataset corresponding to current node.
        @param parent: An integer to represent the parent node of the current node.
        @param even: A boolean to represent whether the current node is at even layer or odd layer.
        """
        self.plot = plot
        self.index = index
        self.parent = parent
        self.even = even


class CART(Algo):

    def fit(self, ccp_alpha, random_state=None):
        """
        This is a wrapper function for DecisionTreeClassifier.
        This function will fit a decision tree and store the feature importance.

        :param ccp_alpha: A floating point that is the hyperparameter of the DecisionTreeClassifier.
        :param random_state: An integer to specify the random state.
        """
        if random_state is None:
            # if no random_state is specified, default to instance's random_state.
            random_state = self.random_state

        # fit the decision tree
        clf = DecisionTreeClassifier(ccp_alpha=ccp_alpha, random_state=random_state)
        clf.fit(self.data, self.label)

        # store the feature importance score
        self.feature_importance = pd.DataFrame(clf.feature_importances_,
                                               index=self.data.columns,
                                               columns=['importance'])
        self.feature_importance = self.feature_importance.sort_values('importance',
                                                                      ascending=False)

    def generate_tree1(self, ccp_alpha, k, max_depth, min_impurity_decrease, random_state=None, generate_plot=False):
        """
        This function will generate a decision using only k features. This function will first run CART once and
        select the top k features in the returned feature importance. Then the function will generate a new dataframe
        with only top k features and use this dataframe to generate a decision tree.

        :param ccp_alpha: A floating point to control to what degree the internal node will be collapsed.
        :param k: An integer to specify the number of features.
        :param max_depth: An integer to specify the maximum depth of the decision tree.
        :param min_impurity_decrease: An integer to control the complexity of the decision tree.
        :param random_state: An integer to specify the random state.
        """

        if random_state is None:
            # if no random_state is specified, default to instance's random_state.
            random_state = self.random_state

        # invoke the wrapper function.
        self.fit(0, random_state=random_state)
        # select top k-features
        k_features = self.k_features(k)
        if len(k_features) < k:
            # print a warning message if less than k features are selected.
            warnings.warn(f'The number of features selected is less than {k}')

        # select k features from the dataset.
        truncated_data = self.data.loc[:, k_features]

        # create a new decision tree using the new dataset.
        clf = DecisionTreeClassifier(ccp_alpha=ccp_alpha,
                                     max_depth=max_depth,
                                     min_impurity_decrease=min_impurity_decrease,
                                     random_state=random_state)
        clf.fit(truncated_data, self.label)

        # store the feature importance
        self.feature_importance = pd.DataFrame(clf.feature_importances_,
                                               index=clf.feature_names_in_,
                                               columns=['importance'])
        # sort the score in descending order
        self.feature_importance = self.feature_importance.sort_values('importance', ascending=False)

        # remove zero-score features
        non_zero_indices = (self.feature_importance.iloc[:, 0] != 0)
        self.feature_importance = self.feature_importance[non_zero_indices]

        # store the tree
        self.tree = clf

        if generate_plot:
            # export the decision tree to a Dot file
            export_graphviz(clf, out_file="tree.dot",
                            feature_names=k_features,
                            class_names=clf.classes_.astype(str),
                            filled=True,
                            rounded=True, special_characters=True)

            # load the dot file and render it
            with open("tree.dot") as f:
                dot_graph = f.read()
            graphviz.Source(dot_graph).view()  # set the size of the plot

    def f(self, ccp_alpha, k, random_state=None):
        """
        This is a helper function that will be invoked by generate_tree2.
        This function is a discrete function.

        :param ccp_alpha: A floating point number.
        :param k: An integer to specify the desired number of features.
        :param random_state: An integer to specify the random state.
        :return: The generated decision tree and an integer number.
        """
        # given ccp_alpha and random_state, fit a decision tree.
        clf = DecisionTreeClassifier(ccp_alpha=ccp_alpha, random_state=random_state)
        clf.fit(self.data, self.label)
        # the number of features whose feature importance is non-zero minus k
        result = np.sum((clf.feature_importances_ != 0)) - k
        return clf, result

    def generate_tree2(self, k, max_itr=1000, random_state=None, generate_plot=False):
        """
        This function will use bisection method to find a decision tree that will exactly use k features.
        This function will treat the task to find the k-feature decision tree as a root finding problem and use helper
            function f to continuously iterate until it finds the desired decision tree.

        :param k: An integer to specify the desired number of features.
        :param max_itr: An integer to specify the maximum number of iterations.
        :param random_state: An integer to specify the random state.
        """
        if random_state is None:
            # if no random_state is specified, default to instance's random_state.
            random_state = self.random_state

        if k > self.data.shape[1] or k <= 0:
            # check k
            # if k is larger than the total amount of features or k is smaller than 0
            # the function terminates
            logging.warning(f'Invalid k {k}')
            return

        # large and small are two pointers that point to two ends of the interval
        large = 1
        large_tree, f_large = self.f(ccp_alpha=large, k=k, random_state=random_state)
        small = 0
        small_tree, f_small = self.f(ccp_alpha=small, k=k, random_state=random_state)
        medium_tree = None
        f_medium = None

        if f_small < 0 or f_large > 0:
            # prevent some extreme cases.
            logging.warning('Unable to find tree, try generate_tree1')
            return

        epoch = 0
        while True:
            # use bisection method to find ccp_alpha

            medium = large - (large - small) / 2
            medium_tree, f_medium = self.f(ccp_alpha=medium, k=k, random_state=random_state)

            # scenario 1:
            if f_medium == 0 or f_large == 0 or f_small == 0:
                # root found
                break

            # scenario 2:
            if f_medium * f_large < 0:
                # continue to iterate
                small = medium
                f_small = f_medium

            # scenario 3:
            if f_medium * f_small < 0:
                # continue to iterate
                large = medium
                f_large = f_medium

            # scenario 4:
            if f_large * f_medium > 0 and f_small * f_medium > 0:
                # Unlikely scenario. The algorithm should converge to one root.
                logging.warning('Failed to converge.')
                break

            # scenario 5:
            if epoch > max_itr:
                logging.warning('Exceed the given maximum iteration.')
                break

            # increment epoch
            epoch = epoch + 1

        # store the tree
        if f_medium == 0:
            self.tree = medium_tree
        elif f_large == 0:
            self.tree = large_tree
        elif f_small == 0:
            self.tree = small_tree
        else:
            logging.warning('Failed to find the tree. Adjust hyperparameter or try generate_tree1.')
            return

        # store the feature importance
        self.feature_importance = pd.DataFrame(self.tree.feature_importances_,
                                               index=self.tree.feature_names_in_,
                                               columns=['importance'])
        # sort the score in descending order
        self.feature_importance = self.feature_importance.sort_values('importance',
                                                                      ascending=False)

        # remove zero score features
        non_zero_indices = (self.feature_importance.iloc[:, 0] != 0)
        self.feature_importance = self.feature_importance[non_zero_indices]

        if generate_plot:
            # export the decision tree to a Dot file
            export_graphviz(self.tree, out_file="tree.dot",
                            feature_names=self.tree.feature_names_in_,
                            class_names=self.tree.classes_.astype(str),
                            filled=True,
                            rounded=True, special_characters=True)

            # load the dot file and render it
            with open("tree.dot") as f:
                dot_graph = f.read()
            graphviz.Source(dot_graph).view()
    @staticmethod
    def split_text(text, max_length):
        """Split text into multiple lines with each line no longer than max_length."""
        words = text.split()
        lines = []
        current_line = [words[0]]
        current_length = len(words[0]) + 1

        for word in words[1:]:

            if current_length + len(word) + 1 <= max_length:
                current_line.append(word)
                current_length += len(word) + 1
            else:

                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word) + 1

        lines.append(' '.join(current_line))

        if len(lines) >= 4:
            final_lines = lines[:1]
            final_lines.append('...')
            final_lines.append(lines[-1])
        else:
            final_lines = lines

        return '\n'.join(final_lines)

    def static_gating_plot(self,
                           y_length,
                           x_length,
                           tree,
                           plot_tree,
                           noise):

        #  initialize a subplot container
        fig, axs = plt.subplots(y_length, x_length, figsize=(x_length * 3 + 4, y_length * 3 + 4))

        # handle a special case: only one node
        if tree.node_count == 1:
            warnings.warn("There is only one node in the decision tree.\nNo separation at all.")
            return

        # general case
        # use parent in PlotNode to backtrack and create plots
        leave_count = 0
        row_labels = []
        for i in range(tree.node_count):
            # iterate through every node
            if tree.children_left[i] == -1 and tree.children_right[i] == -1 and plot_tree[i].plot:
                # this is a leaf
                # set label
                leaf_class = np.argmax(tree.value[i])
                label = self.tree.classes_[leaf_class]
                row_labels.append(label)

                # backtrack until root node
                parent = plot_tree[i].parent
                backtrack_array = [i]
                while parent != 0 and parent != -1:
                    backtrack_array.append(parent)
                    parent = plot_tree[parent].parent
                backtrack_array.append(0)
                backtrack_array = backtrack_array[::-1]

                # path direction from root (left or right)
                # True represents <=
                # False represents >
                directions = []
                for j in range(len(backtrack_array) - 1):
                    node = backtrack_array[j]
                    next_node = backtrack_array[j + 1]
                    if next_node == tree.children_left[node]:
                        directions.append(True)
                    else:
                        directions.append(False)

                # create a series of subplots
                for j, node in enumerate(backtrack_array):
                    plot_node = plot_tree[node]
                    if plot_node.plot:

                        if plot_node.even and (node != backtrack_array[-1]):
                            # hande a special case
                            # there is only one column of plots
                            # if x_length == 1:
                            #     ax = axs[leave_count]
                            # else:
                            ax = axs[leave_count, j // 2]

                            row = leave_count + 1
                            col = (j // 2) + 1
                            
                            node_direction = directions[j]
                            parent_direction = directions[j - 1]

                            # even internal node
                            # extract relevant data
                            parent_node = plot_tree[plot_node.parent]
                            parent_data_index = parent_node.index
                            node_feature = tree.feature[node]
                            parent_node_feature = tree.feature[plot_node.parent]
                            plot_data = self.data.loc[parent_data_index, :]
                            plot_data_node = plot_data.loc[:, self.tree.feature_names_in_[node_feature]]
                            plot_data_parent = plot_data.loc[:, self.tree.feature_names_in_[parent_node_feature]]
                            
                            if noise:
                                # some noise to increase visibility
                                plot_data_node_noise = np.random.normal(0, plot_data_node.std() / 7,
                                                                        size=len(plot_data_node))
                                plot_data_node = plot_data_node + plot_data_node_noise

                                plot_data_parent_noise = np.random.normal(0, plot_data_parent.std() / 7,
                                                                          size=len(plot_data_parent))
                                plot_data_parent = plot_data_parent + plot_data_parent_noise

                            # display the raw datapoints
                            sns.scatterplot(x=plot_data_node, 
                                            y=plot_data_parent,
                                            ax=ax,
                                            color='k', 
                                            s=10)
                
                            # create the plot
                            sns.kdeplot(x=plot_data_node,
                                        y=plot_data_parent,
                                        ax=ax,
                                        fill=True,
                                        bw_adjust=0.6, # Smoothing factor
                                        alpha=0.75,    # Opacity
                                        cmap='coolwarm')
                            
                            ax.set_xlabel(self.tree.feature_names_in_[node_feature])
                            ax.set_ylabel(self.tree.feature_names_in_[parent_node_feature])

                            # highlight the region in the plot
                            x_threshold = tree.threshold[node]
                            y_threshold = tree.threshold[plot_node.parent]

                            x_max = plot_data_node.max()
                            x_min = plot_data_node.min()

                            y_max = plot_data_parent.max()
                            y_min = plot_data_parent.min()
                            
                            ax.set_xlim(x_min - 0.3, x_max + 0.3)
                            ax.set_ylim(y_min - 0.3, y_max + 0.3)

                            if node_direction:
                                # left
                                x_start = x_min
                                x_len = x_threshold - x_min
                            else:
                                # right
                                x_start = x_threshold
                                x_len = x_max - x_start

                            if parent_direction:
                                y_start = y_min
                                y_len = y_threshold - y_min
                            else:
                                y_start = y_threshold
                                y_len = y_max - y_start

                            # create the rectangle and add it to the plot.
                            rect = Rectangle((x_start, y_start), x_len, y_len, edgecolor='red', facecolor='none')
                            ax.add_patch(rect)

                            padding = 1
                            ax.set_xlim([0 - padding, x_max + padding])
                            ax.set_ylim([0 - padding, y_max + padding])

                        elif plot_node.even and (node == backtrack_array[-1]):
                            # even leaf
                            # create the density plot

                            # hande a special case
                            # there is only one column of plots
                            # if x_length == 1:
                            #     ax = axs[leave_count]
                            # else:
                            ax = axs[leave_count, j // 2]

                            node_parent = plot_node.parent
                            parent_direction = directions[j - 1]
                            plot_node = plot_tree[node_parent]
                            data_index = plot_node.index
                            plot_data = self.data.loc[data_index, :]
                            node_feature = tree.feature[node_parent]
                            plot_data_node = plot_data.loc[:, self.tree.feature_names_in_[node_feature]]

                            sns.kdeplot(x=plot_data_node, ax=ax, fill=True, color='blue', bw_adjust=0.2)
                            ax.set_xlabel(self.tree.feature_names_in_[node_feature])

                            # highlight the region
                            x_min = plot_data_node.min()
                            x_max = plot_data_node.max()
                            # y_min = 0
                            y_max = ax.get_ylim()[1]  # get y-axis limits from the density plot

                            x_threshold = tree.threshold[node_parent]

                            if parent_direction:
                                start = x_min
                                x_len = x_threshold - x_min
                            else:
                                start = x_threshold
                                x_len = x_max - start

                            rect = Rectangle((start, 0),
                                             x_len,
                                             y_max,
                                             edgecolor='red',
                                             facecolor='none')
                            ax.add_patch(rect)

                leave_count += 1

        # remove empty subplots
        is_axs_empty = lambda ax: (len(ax.lines) == 0 and len(ax.patches) == 0)
        for i in range(y_length):
            for j in range(x_length):
                if is_axs_empty(axs[i, j]):
                    axs[i, j].axis('off')

        plt.tight_layout(pad=5.0)

        x_label_min = 100

        spacing = 0.01
        if y_length >= 1:
            spacing = (axs[0, 0].get_position().y0 - axs[1, 0].get_position().y1) / 4

        # adding vertical labels
        for i, label in enumerate(row_labels):
            # calculate the center y-position for the current row's label
            ax = axs[i, 0]  # first column of the current row
            bbox = ax.get_position()  # get the bounding box of the current subplot in figure coordinates
            # y = bbox.y0 + bbox.height / 2  # y0 is the bottom, height is the height of the bbox

            # place the text at the calculated y position
            # text = self.split_text(label, 20)
            fig.text(bbox.x0,
                     bbox.y1 + spacing,
                     label,
                     va='bottom',
                     ha='left',
                     fontsize=12,
                     fontweight='bold',
                     transform=fig.transFigure)

        plt.show()

    def interactive_gating_plot(self,
                                y_length,
                                x_length,
                                tree,
                                plot_tree,
                                noise=False):
        # initialize a subplot container
        fig = make_subplots(rows=int(y_length), cols=int(x_length), subplot_titles=[' '] * (x_length * y_length))

        # handle a special case: only one node
        if tree.node_count == 1:
            warnings.warn("There is only one node in the decision tree.\nNo separation at all.")
            return None

        # general case
        # use parent in PlotNode to backtrack and create plots
        leave_count = 0
        row_labels = []
        for i in range(tree.node_count):
            # Iterate through every node
            if tree.children_left[i] == -1 and tree.children_right[i] == -1 and plot_tree[i].plot: # Add condition to check for nodes with None
                # this is a leaf
                # set label
                leaf_class = np.argmax(tree.value[i])
                label = self.tree.classes_[leaf_class]
                row_labels.append(label)

                # backtrack until root node
                parent = plot_tree[i].parent
                backtrack_array = [i]
                while parent != 0 and parent != -1:
                    backtrack_array.append(parent)
                    parent = plot_tree[parent].parent
                backtrack_array.append(0)
                backtrack_array = backtrack_array[::-1]

                # path direction from root (left or right)
                # true represents <=
                # false represents >
                directions = []
                for j in range(len(backtrack_array) - 1):
                    node = backtrack_array[j]
                    next_node = backtrack_array[j + 1]
                    if next_node == tree.children_left[node]:
                        directions.append(True)
                    else:
                        directions.append(False)

                # create a series of subplots
                for j, node in enumerate(backtrack_array):
                    plot_node = plot_tree[node]
                    if plot_node.plot:

                        if plot_node.even and (node != backtrack_array[-1]):
                            # handle a special case
                            # there is only one column of plots
                            # if x_length == 1:
                            #     row = leave_count + 1
                            #     col = 1
                            # else:
                            row = leave_count + 1
                            col = (j // 2) + 1

                            node_direction = directions[j]
                            parent_direction = directions[j - 1]

                            # even internal node
                            # extract relevant data
                            parent_node = plot_tree[plot_node.parent]
                            parent_data_index = parent_node.index
                            node_feature = tree.feature[node]
                            parent_node_feature = tree.feature[plot_node.parent]
                            plot_data = self.data.loc[parent_data_index, :]
                            plot_data_node = plot_data.loc[:, self.tree.feature_names_in_[node_feature]]
                            plot_data_parent = plot_data.loc[:, self.tree.feature_names_in_[parent_node_feature]]

                            if noise:
                                # add some noise to increase visibility
                                plot_data_node_noise = np.random.normal(0,
                                                                        plot_data_node.std() / 7,
                                                                        size=len(plot_data_node))
                                plot_data_node = plot_data_node + plot_data_node_noise

                                plot_data_parent_noise = np.random.normal(0,
                                                                          plot_data_parent.std() / 7,
                                                                          size=len(plot_data_parent))
                                plot_data_parent = plot_data_parent + plot_data_parent_noise

                            colorscale = [
                                [0, 'rgb(255,255,255)'], [0.25, 'rgb(0,0,100)'],
                                [0.5, 'rgb(0,0,255)'], [0.75, 'rgb(255, 0, 255)'],
                                [1, 'rgb(255,0,0)']
                            ]

                            df = pd.DataFrame({self.tree.feature_names_in_[node_feature]: plot_data_node,
                                               self.tree.feature_names_in_[parent_node_feature]: plot_data_parent})

                            fig.add_trace(go.Scatter(
                                                x=plot_data_node,
                                                y=plot_data_parent,
                                                mode='markers',
                                                opacity=0.1
                                                ), row=row, col=col)

                            custom_colorscale = [
                                [0, 'rgba(0, 0, 255, 0)'],  # Transparent at the lower end
                                [1, 'rgba(0, 0, 255, 1)']  # Blue at the upper end
                            ]

                            fig.add_trace(go.Histogram2dContour(
                                x=plot_data_node,
                                y=plot_data_parent,
                                colorscale=custom_colorscale,
                                showscale=False
                            ), row=row, col=col)

                            fig.update_xaxes(title_text=self.tree.feature_names_in_[node_feature],
                                             row=row,
                                             col=col)

                            fig.update_yaxes(title_text=self.tree.feature_names_in_[parent_node_feature],
                                             row=row,
                                             col=col)

                            # set the y-axis range explicitly
                            fig.update_yaxes(range=[plot_data_parent.min() - 1, plot_data_parent.max() + 1],
                                             row=row,
                                             col=col)

                            # highlight the region in the plot
                            x_threshold = tree.threshold[node]
                            y_threshold = tree.threshold[plot_node.parent]

                            x_max = plot_data_node.max()
                            x_min = plot_data_node.min()

                            y_max = plot_data_parent.max()
                            y_min = plot_data_parent.min()

                            if node_direction:
                                # left
                                x_start = x_min
                                x_len = x_threshold - x_min
                            else:
                                # right
                                x_start = x_threshold
                                x_len = x_max - x_start

                            if parent_direction:
                                y_start = y_min
                                y_len = y_threshold - y_min
                            else:
                                y_start = y_threshold
                                y_len = y_max - y_start

                            # create the rectangle and add it to the plot
                            fig.add_shape(
                                type="rect",
                                x0=x_start, y0=y_start, x1=x_start + x_len, y1=y_start + y_len,
                                line=dict(color="Red", width=2),
                                row=row, col=col
                            )

                        elif plot_node.even and (node == backtrack_array[-1]):
                            # even leaf
                            # create the density plot
                            # if x_length == 1:
                            #     row = leave_count + 1
                            #     col = 1
                            # else:
                            row = leave_count + 1
                            col = (j // 2) + 1

                            node_parent = plot_node.parent
                            parent_direction = directions[j - 1]
                            plot_node = plot_tree[node_parent]
                            data_index = plot_node.index
                            plot_data = self.data.loc[data_index, :]
                            node_feature = tree.feature[node_parent]
                            plot_data_node = plot_data.loc[:, self.tree.feature_names_in_[node_feature]]

                            # prepare data for distplot
                            data_for_distplot = plot_data_node.values

                            # create distplot
                            distplot_fig = ff.create_distplot(
                                [data_for_distplot],
                                [self.tree.feature_names_in_[node_feature]],
                                show_hist=False,
                                show_rug=False
                            )

                            # add KDE trace
                            fig.add_trace(
                                distplot_fig.data[0],
                                row=row, col=col
                            )

                            fig.update_xaxes(title_text=self.tree.feature_names_in_[node_feature], row=row, col=col)

                            # extract the y-values (density values) from the plot
                            y_values = []
                            for trace in distplot_fig['data']:
                                if 'y' in trace:
                                    y_values.extend(trace['y'])

                            # set the y-axis range explicitly
                            y_max = max(y_values)

                            # highlight the region
                            x_min = plot_data_node.min()
                            x_max = plot_data_node.max()
                            y_min = 0

                            x_threshold = tree.threshold[node_parent]

                            if parent_direction:
                                start = x_min
                                x_len = x_threshold - x_min
                            else:
                                start = x_threshold
                                x_len = x_max - start

                            fig.add_shape(
                                type="rect",
                                x0=start, y0=y_min, x1=start + x_len, y1=y_max,
                                line=dict(color="Red", width=2),
                                row=row, col=col
                            )

                leave_count += 1

        # finalize the figure
        fig.update_layout(height=600 * y_length, width=600 * x_length, showlegend=False)

        # adding vertical labels
        for i, label in enumerate(row_labels):
            fig.layout.annotations[i * x_length].update(text=label)

        pyo.plot(fig, filename='multiple_plots.html', auto_open=True)

    def create_dash_gating_plot_app(self,
                                    y_length,
                                    x_length,
                                    tree,
                                    plot_tree,
                                    noise=False):

        # initialize a 2D array to store plotly figures
        plotly_figs = [[None for _ in range(x_length)] for _ in range(y_length)]

        # handle a special case: only one node
        if tree.node_count == 1:
            warnings.warn("There is only one node in the decision tree.\nNo separation at all.")
            exit(0)

        # create a collection of plots and store in an array
        # use parent in PlotNode to backtrack and create plots
        leave_count = 0
        row_labels = []
        label_index = 1
        for i in range(tree.node_count):
            # iterate through every node
            if tree.children_left[i] == -1 and tree.children_right[i] == -1 and plot_tree[i].plot: # Add condition to check for nodes with None
                # this is a leaf
                # set label
                leaf_class = np.argmax(tree.value[i])
                label = self.tree.classes_[leaf_class]
                row_labels.append(f"{label_index}. {label}")
                label_index += 1

                # backtrack until root node
                parent = plot_tree[i].parent
                backtrack_array = [i]
                while parent != 0 and parent != -1:
                    backtrack_array.append(parent)
                    parent = plot_tree[parent].parent
                backtrack_array.append(0)
                backtrack_array = backtrack_array[::-1]

                # path direction from root (left or right)
                # true represents <=
                # false represents >
                directions = []
                for j in range(len(backtrack_array) - 1):
                    node = backtrack_array[j]
                    next_node = backtrack_array[j + 1]
                    if next_node == tree.children_left[node]:
                        directions.append(True)
                    else:
                        directions.append(False)

                # create a series of subplots
                for j, node in enumerate(backtrack_array):
                    plot_node = plot_tree[node]
                    if plot_node.plot:

                        if plot_node.even and (node != backtrack_array[-1]):
                            # handle a special case
                            # there is only one column of plots
                            # if x_length == 1:
                            #     row = leave_count + 1
                            #     col = 1
                            # else:
                            row = leave_count
                            col = (j // 2)

                            node_direction = directions[j]
                            parent_direction = directions[j - 1]

                            # even internal node
                            # extract relevant data
                            parent_node = plot_tree[plot_node.parent]
                            parent_data_index = parent_node.index
                            node_feature = tree.feature[node]
                            parent_node_feature = tree.feature[plot_node.parent]
                            plot_data = self.data.loc[parent_data_index, :]
                            plot_data_node = plot_data.loc[:, self.tree.feature_names_in_[node_feature]]
                            plot_data_parent = plot_data.loc[:, self.tree.feature_names_in_[parent_node_feature]]

                            if noise:
                                # add some noise to increase visibility
                                plot_data_node_noise = np.random.normal(0,
                                                                        plot_data_node.std() / 7,
                                                                        size=len(plot_data_node))
                                plot_data_node = plot_data_node + plot_data_node_noise

                                plot_data_parent_noise = np.random.normal(0,
                                                                          plot_data_parent.std() / 7,
                                                                          size=len(plot_data_parent))
                                plot_data_parent = plot_data_parent + plot_data_parent_noise

                            # colorscale = ['#7A4579', '#D56073', 'rgb(236,158,105)', (1, 1, 0.2), (0.98, 0.98, 0.98)]
                            colorscale = [(1, 0, 0), (0.5, 0.5, 0), (0, 1, 0), (0, 0.5, 0.5), (1, 1, 1)]

                            x_axis_title = self.tree.feature_names_in_[node_feature]
                            y_axis_title = self.tree.feature_names_in_[parent_node_feature]

                            df = pd.DataFrame({x_axis_title: plot_data_node,
                                               y_axis_title: plot_data_parent})

                            plot = ff.create_2d_density(
                                df.iloc[:, 0], df.iloc[:, 1], colorscale=colorscale,
                                hist_color='rgb(255, 237, 222)', point_size=3,
                            )

                            plot.update_layout(
                                coloraxis=dict(
                                    colorscale=colorscale,
                                    colorbar=dict(
                                        title='Density',
                                        titleside='right'
                                    )
                                ),
                                xaxis_title=x_axis_title,
                                yaxis_title=y_axis_title,
                                title={
                                    'text': f'{x_axis_title} - {y_axis_title} Gating Plot',
                                    'x': 0.5,  # this centers the title
                                    'xanchor': 'center'
                                }
                            )

                            colorbar_trace = go.Scatter(x=[None],
                                                        y=[None],
                                                        mode='markers',
                                                        marker=dict(
                                                            colorscale=colorscale[::-1],
                                                            showscale=True,
                                                            cmin=0,
                                                            cmax=5,
                                                            colorbar=dict(thickness=5, tickvals=[0, 5],
                                                                          ticktext=['Low', 'High'], outlinewidth=0)
                                                        ),
                                                        hoverinfo='none'
                                                        )

                            plot['layout']['showlegend'] = False
                            plot.add_trace(colorbar_trace)

                            plotly_figs[row][col] = plot

                            # highlight the region in the plot
                            x_threshold = tree.threshold[node]
                            y_threshold = tree.threshold[plot_node.parent]

                            x_max = plot_data_node.max()
                            x_min = plot_data_node.min()

                            y_max = plot_data_parent.max()
                            y_min = plot_data_parent.min()

                            if node_direction:
                                # left
                                x_start = x_min
                                x_len = x_threshold - x_min
                            else:
                                # right
                                x_start = x_threshold
                                x_len = x_max - x_start

                            if parent_direction:
                                y_start = y_min
                                y_len = y_threshold - y_min
                            else:
                                y_start = y_threshold
                                y_len = y_max - y_start

                            # create the rectangle and add it to the plot
                            plot.add_shape(
                                type="rect",
                                x0=x_start, y0=y_start, x1=x_start + x_len, y1=y_start + y_len,
                                line=dict(color="Red", width=2)
                            )

                        elif plot_node.even and (node == backtrack_array[-1]):
                            # even leaf
                            # create the density plot
                            # if x_length == 1:
                            #     row = leave_count + 1
                            #     col = 1
                            # else:
                            row = leave_count
                            col = (j // 2)

                            node_parent = plot_node.parent
                            parent_direction = directions[j - 1]
                            plot_node = plot_tree[node_parent]
                            data_index = plot_node.index
                            plot_data = self.data.loc[data_index, :]
                            node_feature = tree.feature[node_parent]
                            plot_data_node = plot_data.loc[:, self.tree.feature_names_in_[node_feature]]

                            # prepare data for distplot
                            data_for_distplot = plot_data_node.values

                            # create distplot
                            plot = ff.create_distplot(
                                [data_for_distplot],
                                [self.tree.feature_names_in_[node_feature]],
                                show_hist=False,
                                show_rug=False
                            )

                            plot.update_layout(xaxis_title=self.tree.feature_names_in_[node_feature],
                                               yaxis_title='Probability Density',
                                               showlegend=False,
                                               title={
                                                   'text': f'{self.tree.feature_names_in_[node_feature]} Distribution Plot',
                                                   'x': 0.5,  # This centers the title
                                                   'xanchor': 'center'
                                               })

                            plotly_figs[row][col] = plot

                            # extract the y-values (density values) from the plot
                            y_values = []
                            for trace in plot['data']:
                                if 'y' in trace:
                                    y_values.extend(trace['y'])

                            # find the min and max probability density values
                            y_min = 0
                            y_max = max(y_values)

                            x_min = plot_data_node.min()
                            x_max = plot_data_node.max()

                            x_threshold = tree.threshold[node_parent]

                            if parent_direction:
                                start = x_min
                                x_len = x_threshold - x_min
                            else:
                                start = x_threshold
                                x_len = x_max - start

                            plot.add_shape(
                                type="rect",
                                x0=start, y0=y_min, x1=start + x_len, y1=y_max,
                                line=dict(color="Red", width=2)
                            )

                leave_count += 1

        # initialize an instance
        app = dash.Dash(__name__,
                        external_stylesheets=[dbc.themes.COSMO],
                        meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}])

        app.layout = dbc.Container([
            html.Br(),

            # headline
            dbc.Row([
                dbc.Col(
                    html.H1('Gating Plot Web App',
                            className='text-center text-primary mb-4'),
                    width=12
                )
            ], justify='center'),
            dbc.Row([
                dbc.Tooltip("Select a cell type from the dropdown",
                            target="dropdown"),
                dcc.Dropdown(row_labels,
                             row_labels[0],
                             id='dropdown'),
            ]),
            html.Br(),
            html.Br(),
            html.Br(),
            # select
            dbc.Row([
                html.Div([
                    dbc.Tooltip(
                        "Select marker(s) from the slider",
                        target="slider"
                    ),
                    dcc.Slider(min=0,
                               max=1,
                               id='slider',
                               step=None, )
                ], id='slider-div')
            ], justify='center'),
            html.Br(),
            # show the plot
            # show the plot
            dbc.Tooltip(
                "Select a region to zoom in. Double click to zoom out.",
                target="graph-container"
            ),
            dbc.Row([
                dbc.Col(
                    [dcc.Graph(id='graph', figure={})],
                    width={"size": 8, "offset": 2}  # adjust width and offset as needed
                )
            ], justify='center', id='graph-container')
        ])

        @app.callback(
            [Output('slider', 'max'),
             Output('slider', 'marks'),
             Output('slider-div', 'style'),
             Output('slider', 'value')],
            Input('dropdown', 'value')
        )
        def update_slider(label=row_labels[0]):

            # acquire the row index
            index = 0
            for i, l in enumerate(row_labels):
                if l == label:
                    index = i
                    break

            plot_labels = []
            for plot in plotly_figs[index]:
                if plot is not None:
                    x_label = plot.layout.xaxis.title.text
                    y_label = plot.layout.yaxis.title.text
                    if y_label == 'Probability Density':
                        strg = x_label
                    else:
                        strg = f'{x_label} {y_label}'
                    plot_labels.append(strg.strip())

            max_val = len(plot_labels) - 1
            marks = dict()
            for i, l in enumerate(plot_labels):
                marks[i] = l

            if max_val == 0:
                return 0, dict(), {'display': 'none'}, 0

            return max_val, marks, {'display': 'block'}, 0

        @app.callback(
            Output('graph', 'figure'),
            [Input('dropdown', 'value'),
             Input('slider', 'value')]
        )
        def update_graph(protein_label=row_labels[0], plot_index=0):

            # acquire the row index
            index = 0
            for i, l in enumerate(row_labels):
                if l == protein_label:
                    index = i
                    break

            if protein_label is None:
                protein_label = 0
            if plot_index is None:
                plot_index = 0

            fig = plotly_figs[index][plot_index]

            return fig

        return app

    def generate_gating_plot(self, noise=False, plot_option=1):
        """
        This function will generate a region plot of decision trees.

        @param noise: A boolean to indicate whether to add noise or not.
            Points tend to lay together.
            The added random noise can increase the visibility.
        """
        tree = self.tree.tree_

        # this array will store plots as the function iterates through the decision tree.
        plot_tree = [PlotNode(False, None, None, None) for _ in range(len(tree.children_left))]

        # initialize some variables
        node_stack = [0]  # the main stack, represents the node
        parent_stack = [-1]  # a stack to represent current node's parent
        # a stack used to store the current node's corresponding dataset index
        data_index_stack = [self.data.index]
        even_stack = [False]  # a stack to represent whether the current node is in odd layer or even layer.

        # preorder traversal with stack
        while len(node_stack) > 0:
            node = node_stack.pop()
            parent = parent_stack.pop()
            data_index = data_index_stack.pop()
            even = even_stack.pop()

            # process node
            plot_tree[node].index = data_index
            plot_tree[node].parent = parent
            plot_tree[node].even = even

            if not even:

                # 1. odd layer leaf node
                if tree.children_right[node] == -1 and tree.children_left[node] == -1:
                    # this is the leaf node
                    plot_tree[node].plot = True

                else:
                    # 2. odd layer internal node
                    plot_tree[node].plot = False

            else:
                # even layer leaf or internal nodes
                plot_tree[node].plot = True

            feature = tree.feature[node]
            feature_name = self.tree.feature_names_in_[feature]
            node_data = self.data.loc[data_index, :]
            node_feature_data = node_data.loc[:, feature_name]
            threshold = tree.threshold[node]
            if tree.children_right[node] != -1:  # push right child to stack if there is one
                node_stack.append(tree.children_right[node])
                parent_stack.append(node)
                even_stack.append(not even)
                # process and append data index
                # > threshold
                larger_than_threshold = node_feature_data.index[np.where(node_feature_data > threshold)]
                data_index_stack.append(larger_than_threshold)

            if tree.children_left[node] != -1:  # push left child to stack if there is one
                node_stack.append(tree.children_left[node])
                parent_stack.append(node)
                even_stack.append(not even)
                # process and append data index
                # <= threshold
                smaller_than_threshold = node_feature_data.index[np.where(node_feature_data <= threshold)]
                data_index_stack.append(smaller_than_threshold)

        # prepare for plotting
        # to get the depth of the tree
        tree_depth = self.tree.get_depth() + 1
        x_length = math.floor(tree_depth / 2)
        y_length = self.tree.get_n_leaves()
        if x_length == 1:
            # x_length = 1 may cause some problems so add 1
            x_length = 2

        # invoke helper function to generate the plot
        if plot_option == 1:
            self.static_gating_plot(y_length, x_length, tree, plot_tree, noise)
        elif plot_option == 2:
            self.interactive_gating_plot(y_length, x_length, tree, plot_tree, noise)
        elif plot_option == 3:
            print("Please visit http://127.0.0.1:8050 to view the plot")
            app = self.create_dash_gating_plot_app(y_length, x_length, tree, plot_tree, noise)
            app.run(jupyter_mode="_none", host="0.0.0.0", port=8050, debug=False)

    def __init__(self, data, label):

        super().__init__()
        self.tree = None
        self.data = data
        self.label = label