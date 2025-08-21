import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from const import Const
import numpy as np
import os
import statistics
import pandas as pd
from statsmodels.graphics.mosaicplot import mosaic
import plotly.graph_objects as go
import seaborn as sns
from collections import Counter
from divergence_helper import js_divergence
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import MinMaxScaler

class Plot_Helper:
    """
    Helper class for creating various plots for facial feature analysis.
    """

    gen_dir = '../results/plots'
    plot_dir = ''

    def __init__(self, date: str):
        """
        Initializes the Plot_Helper with a directory for saving plots.

        Args:
            date (str): The date string used to create a subdirectory for plots.
        """
        self.plot_dir = os.path.join(self.gen_dir, date)
        os.makedirs(self.plot_dir, exist_ok=True)

    def __prepare_data_for_dist_plotting(self, df: pd.DataFrame, feature: str, group_by: str = None) -> pd.DataFrame:
        """
        Prepares data in a long-form DataFrame suitable for seaborn plotting.
        Calculates both observed and normative (uniform) distributions.

        Args:
            df (pd.DataFrame): The input data.
            feature (str): The feature to plot.
            group_by (str, optional): The feature to group by for conditional distributions.

        Returns:
            pd.DataFrame: DataFrame ready for plotting.
        """
        # Define bins for age, treat others as categorical
        # Determine the grouping columns
        grouping_cols = [group_by, feature] if group_by else [feature]

        # Calculate observed counts and probabilities
        observed_counts = df.groupby(grouping_cols).size().reset_index(name='count')
        if group_by:
            total_in_group = observed_counts.groupby(group_by)['count'].transform('sum')
        else:
            total_in_group = observed_counts['count'].sum()

        observed_probs = observed_counts.copy()
        observed_probs['probability'] = observed_probs['count'] / total_in_group
        observed_probs['Distribution'] = 'Observed'

        # Calculate normative (uniform) probabilities
        normative_probs = observed_probs.copy()
        if group_by:
            # For each group, the normative probability is 1 / (number of categories in that group)
            num_categories = normative_probs.groupby(group_by)[feature].transform('nunique')
            normative_probs['probability'] = 1 / num_categories
        else:
            # For the overall distribution, it's 1 / (total number of categories)
            num_categories = normative_probs[feature].nunique()
            normative_probs['probability'] = 1 / num_categories

        normative_probs['Distribution'] = 'Normative'

        # Combine into a single long-form DataFrame
        plot_df = pd.concat([observed_probs, normative_probs], ignore_index=True)

        return plot_df.rename(columns={feature: 'Category'})

    def create_race_gender_mosaic_plot(self, cobined_dict: dict):
        """
        Creates a mosaic plot to visualize intersectional bias between race and gender.

        Args:
            cobined_dict (dict): Dictionary containing facial feature data.
        """
        df = pd.DataFrame(cobined_dict)

        # Create a contingency table
        contingency_table = pd.crosstab(df[Const.race], df[Const.gender])

        plt.style.use('default') # Mosaic plot styles itself
        fig, ax = plt.subplots(figsize=(14, 10))

        # The 'props' function defines how to color tiles based on residuals
        props = lambda key: {'color': '#d62728' if contingency_table.loc[key] < 0 else '#2ca02c'}

        mosaic(contingency_table.stack(), ax=ax, title="title", gap=0.02)

        # Customize labels and title
        ax.set_title("Test", fontsize=16, pad=40)
        ax.set_xlabel("Race", fontsize=12)
        ax.set_ylabel("Gender", fontsize=12)

        # Add legend manually
        red_patch = mpatches.Patch(color='#d62728', label='Observed < Expected (Under-represented)')
        green_patch = mpatches.Patch(color='#2ca02c', label='Observed > Expected (Over-represented)')
        plt.legend(handles=[red_patch, green_patch], bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(os.path.join(self.plot_dir, "filename.svg"))
        print("Saved intersectional_mosaic_plot.svg")
        plt.close('all')

    def _create_heatmap(self, data_df: pd.DataFrame, values: str, index: str, columns: str, filename: str, title: str, xlabel: str, ylabel: str):
        """
        Internal method to create a heatmap from a DataFrame.

        Args:
            data_df (pd.DataFrame): Data for plotting.
            values (str): Value column for heatmap.
            index (str): Row categories.
            columns (str): Column categories.
            filename (str): Output filename.
            title (str): Plot title.
            xlabel (str): X-axis label.
            ylabel (str): Y-axis label.
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure()

        pivot_df = data_df.pivot_table(
            values=values,
            index=index,
            columns=columns,
            aggfunc=[np.std, np.mean]
        )

        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".1f",
            cmap='viridis',
            linewidths=.5
        )

        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.xticks(rotation=45, ha='right')
        # plt.xticks(rotation=45, ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, filename))
        print(f'created plot: {os.path.join(self.plot_dir, filename)}')
        plt.close('all')

    def _create_boxplots(self, data_df: pd.DataFrame, xvalues: str, yvalues: str, filename: str, title: str, xlabel: str, ylabel: str):
        """
        Internal method to create boxplots.

        Args:
            data_df (pd.DataFrame): Data for plotting.
            xvalues (str): X-axis feature.
            yvalues (str): Y-axis feature.
            filename (str): Output filename.
            title (str): Plot title.
            xlabel (str): X-axis label.
            ylabel (str): Y-axis label.
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure()

        # Custom sort key for age_range
        if xvalues == Const.age_range:
            order = sorted(data_df[xvalues].unique(), key=lambda x: int(x.split('-')[0]))
            print(order)
        else:
            order = None

        sns.boxplot(data=data_df, x=xvalues, y=yvalues, palette="mako", orient="v", order=order)

        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.xticks(rotation=45, ha='right')
        # plt.xticks(rotation=45, ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, filename))
        print(f'created plot: {os.path.join(self.plot_dir, filename)}')
        plt.close('all')

    def create_boxplot(self, _gen_dict: dict, x_type: str, y_type: str):
        """
        Creates and saves a boxplot for the given features.

        Args:
            _gen_dict (dict): Data dictionary.
            x_type (str): X-axis feature.
            y_type (str): Y-axis feature.
        """
        df = pd.DataFrame(_gen_dict)
        scaler = MinMaxScaler()
        df['mean_depth'] = scaler.fit_transform(df[['mean_depth']])
        self._create_boxplots(data_df=df, xvalues=x_type, yvalues=y_type, ylabel=y_type, filename=f'{x_type}_{y_type}_boxplot.svg', title=f'{y_type}s distribution {x_type}', xlabel=x_type)

    def create_heatmap(self, _gen_dict: dict, values: str, index: str, columns: str):
        """
        Creates and saves a heatmap for the given features.

        Args:
            _gen_dict (dict): Data dictionary.
            values (str): Value column for heatmap.
            index (str): Row categories.
            columns (str): Column categories.
        """
        df = pd.DataFrame(_gen_dict)
        self._create_heatmap(data_df=df, values=values, index=index, columns=columns, ylabel=index, filename=f'{index}_{columns}_{values}_heatmap.svg', title=f'average {values} of {columns} and {index} combined', xlabel=columns)

    def create_multidimensional_heatmap(self, _gen_dict: dict, values: str, index: str, columns: str, grouping: str, low_count_threshold: int = 20):
        """
        Creates a multidimensional heatmap faceted by a grouping variable.

        Args:
            _gen_dict (dict): Data dictionary.
            values (str): Value column for heatmap.
            index (str): Row categories.
            columns (str): Column categories.
            grouping (str): Facet variable.
            low_count_threshold (int): Threshold for low sample size indication.
        """
        data_df = pd.DataFrame(_gen_dict)

        # Define the order of categories for consistent plotting
        index_order = [r for r in Const.possible_values[index] if r in data_df[index].unique()]
        columns_order = [g for g in Const.possible_values[columns] if g in data_df[columns].unique()]
        grouping_order = sorted(data_df[grouping].unique(), key=lambda x: int(x.split('-')[0]) if '-' in x else x) # Custom sort key for age_range

        global_mean_min = data_df.groupby([index, columns, grouping])[values].mean().min()
        global_mean_max = data_df.groupby([index, columns, grouping])[values].mean().max()

        def draw_heatmap(*args, **kwargs):
            """A helper function to draw a heatmap on a FacetGrid axis."""
            # This function receives a subset of the data for each facet
            data = kwargs.pop('data')
            # Create the pivot table for the subset of data
            pivot = data.pivot_table(values=values, index=index, columns=columns, aggfunc=['mean', 'size'], fill_value=0)

            multi_cols = pd.MultiIndex.from_product([['mean', 'size'], columns_order], names=['aggfunc', columns])

            # Reindex to ensure all categories are present and in order, filling missing with NaN
            pivot = pivot.reindex(index=index_order, columns=multi_cols, fill_value=0).astype(float)

            # Separate the mean values (for color) and size values (for annotation)
            mean_data = pivot['mean']
            size_data = pivot['size']

            # Create the custom annotation strings
            mean_strings = mean_data.stack().apply(lambda x: f"Pos: {x:.2f}\n").unstack()
            size_strings = size_data.stack().apply(lambda x: f"N: {int(x)}").unstack()

            annot_labels = mean_strings + size_strings

            # Draw the heatmap on the current axis
            ax = plt.gca()
            sns.heatmap(
                mean_data,
                annot=annot_labels,
                fmt="",  # Use empty fmt because annotations are already strings
                cmap='plasma',
                linewidths=.5,
                ax=ax,
                cbar=True, # Add a color bar to each subplot for clarity
                cbar_kws={'label': 'Avg. Face Position'},
                vmin=global_mean_min, # Set minimum for color scale
                vmax=global_mean_max  # Set maximum for color scale
            )

                    # Add visual indication for low counts (border)
            for i in range(len(index_order)):
                for j in range(len(columns_order)):
                    ind = index_order[i]
                    column = columns_order[j]
                    count = size_data.loc[ind, column]
                    # Only add border if count is greater than 0 and below threshold
                    if count > 0 and count < low_count_threshold:
                        # Add a border to the cell
                        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='grey', lw=2, hatch='///'))
                    # Also add a border for cells with count 0 for consistency if desired,
                    # though the annotation already shows N: 0
                    elif count == 0:
                        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=1, hatch='///')) # Using gray border for 0 count

        # Use seaborn's FacetGrid to create a grid of plots faceted by 'emotion'
        # col_wrap=3 means there will be a maximum of 3 plots per row.
        g = sns.FacetGrid(data_df, col=grouping, col_order=grouping_order, aspect=1, col_wrap=3, height=6,)

        # Map the heatmap drawing function to each facet in the grid.
        # We pass vmin and vmax to ensure a consistent color scale across all plots.
        g.map_dataframe(draw_heatmap, columns, index, values, vmin=0, vmax=11)
        g.set_titles(col_template="{col_name}") # Set subplot titles to the emotion name

        g.figure.suptitle(f'Average {values} by {columns}, {index}, and {grouping}', fontsize=20)
        g.tick_params(axis='x', which='both', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f'{columns}_{values}_{index}_{grouping}_heatmap.svg'))
        print(f'created plot: {os.path.join(self.plot_dir, f'{columns}_{values}_{index}_{grouping}_heatmap.svg')}')
        plt.close('all')

    def create_multi_multidimensional_heatmap(self, _gen_dict: dict, values: str, index: str, columns: str, grouping: str, major_grouping: str, low_count_threshold: int = 20):
        """
        Creates a multi-multidimensional heatmap faceted by two grouping variables.

        Args:
            _gen_dict (dict): Data dictionary.
            values (str): Value column for heatmap.
            index (str): Row categories.
            columns (str): Column categories.
            grouping (str): Row facet variable.
            major_grouping (str): Column facet variable.
            low_count_threshold (int): Threshold for low sample size indication.
        """
        data_df = pd.DataFrame(_gen_dict)

        # Define the order of categories for consistent plotting
        index_order = [r for r in Const.possible_values[index] if r in data_df[index].unique()]
        columns_order = [g for g in Const.possible_values[columns] if g in data_df[columns].unique()]
        grouping_order = sorted(data_df[grouping].unique(), key=lambda x: int(x.split('-')[0]) if '-' in x else x) # Custom sort key for age_range
        major_grouping_order = sorted(data_df[major_grouping].unique(), key=lambda x: int(x.split('-')[0]) if '-' in x else x) # Custom sort key for age_range

        global_mean_min = data_df.groupby([index, columns, grouping])[values].mean().min()
        global_mean_max = data_df.groupby([index, columns, grouping])[values].mean().max()

        def draw_heatmap(*args, **kwargs):
            """A helper function to draw a heatmap on a FacetGrid axis."""
            # This function receives a subset of the data for each facet
            data = kwargs.pop('data')
            # Create the pivot table for the subset of data
            pivot = data.pivot_table(values=values, index=index, columns=columns, aggfunc=['mean', 'size'], fill_value=0)

            multi_cols = pd.MultiIndex.from_product([['mean', 'size'], columns_order], names=['aggfunc', columns])

            # Reindex to ensure all categories are present and in order, filling missing with NaN
            pivot = pivot.reindex(index=index_order, columns=multi_cols, fill_value=0).astype(float)

            # Separate the mean values (for color) and size values (for annotation)
            mean_data = pivot['mean']
            size_data = pivot['size']

            # Create the custom annotation strings
            mean_strings = mean_data.stack().apply(lambda x: f"Pos: {x:.2f}\n").unstack()
            size_strings = size_data.stack().apply(lambda x: f"N: {int(x)}").unstack()

            annot_labels = mean_strings + size_strings

            # Draw the heatmap on the current axis
            ax = plt.gca()
            sns.heatmap(
                mean_data,
                annot=annot_labels,
                fmt="",  # Use empty fmt because annotations are already strings
                cmap='plasma',
                linewidths=.5,
                ax=ax,
                cbar=False, # Add a color bar to each subplot for clarity
                cbar_kws={'label': 'Avg. Face Position'},
                vmin=global_mean_min, # Set minimum for color scale
                vmax=global_mean_max  # Set maximum for color scale
            )

                    # Add visual indication for low counts (border)
            for i in range(len(index_order)):
                for j in range(len(columns_order)):
                    ind = index_order[i]
                    column = columns_order[j]
                    count = size_data.loc[ind, column]
                    # Only add border if count is greater than 0 and below threshold
                    if count > 0 and count < low_count_threshold:
                        # Add a border to the cell
                        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='grey', lw=2, hatch='///'))
                    # Also add a border for cells with count 0 for consistency if desired,
                    # though the annotation already shows N: 0
                    elif count == 0:
                        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=1, hatch='///')) # Using gray border for 0 count

                # Use seaborn's FacetGrid to create a grid of plots faceted by 'emotion'
        # col_wrap=3 means there will be a maximum of 3 plots per row.
        g = sns.FacetGrid(data_df, col=major_grouping, row=grouping, aspect=1, height=6, col_order=major_grouping_order, row_order=grouping_order)

        # Map the heatmap drawing function and collect returned distances
        g.map_dataframe(draw_heatmap)
        g.set_titles(row_template="{row_name}", col_template="{col_name}")
        g.set_axis_labels(columns, index)

        g.figure.suptitle(f'Average {values} by {columns}, {index}, and {grouping}', fontsize=20)
        g.tick_params(axis='x', which='both', rotation=45)

        g.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f'{columns}_{values}_{index}_{grouping}_{major_grouping}_heatmap.svg'))
        print(f'created plot: {os.path.join(self.plot_dir, f'{columns}_{values}_{index}_{grouping}_{major_grouping}_heatmap.svg')}')
        plt.close('all')

    # divergence

    def __create_simpl_distribution_plot(self, observed_dist: dict, normative_dist: dict, labels: list, title: str, filename: str):
        """
        Internal method to create a simple distribution bar plot comparing observed and normative distributions.

        Args:
            observed_dist (dict): Observed distribution counts.
            normative_dist (dict): Normative (uniform) distribution counts.
            labels (list): Category labels.
            title (str): Plot title.
            filename (str): Output filename.
        """
        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))

        # Convert distributions to arrays in the correct order
        observed_values = np.array([observed_dist.get(label, 0) for label in labels])
        normative_values = np.array([normative_dist.get(label, 0) for label in labels])

        # Normalize to probabilities
        observed_probs = observed_values / observed_values.sum() if observed_values.sum() > 0 else observed_values
        normative_probs = normative_values / normative_values.sum() if normative_values.sum() > 0 else normative_values

        rects1 = ax.bar(x - width/2, observed_probs, width, label='Observed', color='skyblue')
        rects2 = ax.bar(x + width/2, normative_probs, width, label='Normative', color='salmon')

        ax.bar_label(rects1, padding=3, fmt='%.3f')
        ax.bar_label(rects2, padding=3, fmt='%.3f')

        ax.set_ylabel('Probability')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        fig.tight_layout()

        plt.savefig(os.path.join(self.plot_dir, filename))
        print(f'created plot: {os.path.join(self.plot_dir, filename)}')
        plt.close('all')

    def create_simpl_distri_plot(self, _dict: dict, type: str):
        """
        Creates a simple distribution plot for a given feature, comparing observed and normative distributions.

        Args:
            _dict (dict): Data dictionary.
            type (str): Feature to plot.
        """
        types = [face[type] for face in _dict]
        type_counts = Counter(types)
        # Custom sort for age_range
        if type == Const.age_range:
             type_labels = sorted(type_counts.keys(), key=lambda x: int(x.split('-')[0]) if '-' in x else x)
        else:
             type_labels = sorted(type_counts.keys())


        observed_type_dist = [type_counts[label] for label in type_labels]

        # Normative distribution: Uniform across observed types
        num_types = len(type_labels)
        normative_type_dist = [1/num_types] * num_types

        js_type = jensenshannon(observed_type_dist, normative_type_dist)**2

        # print(f"Observed type Counts: {dict(type_counts)}")
        # print(f"JS Divergence for type: {js_type:.4f}")

        self.__create_simpl_distribution_plot(
            type_counts,
            {label: sum(observed_type_dist)/num_types for label in type_labels},
            type_labels,
            f'{type} Distribution Comparison, js distance: {js_type:.4f}',
            f'{type}_simpl_distribution_plot.svg'
        )

    def create_comp_distribution_plots_wout_dis(self, _dict: dict, primary_feature: str, conditioning_features: list):
        """
        Creates comparative distribution plots without JS distance (wout_dis).

        Args:
            _dict (dict): Data dictionary.
            primary_feature (str): Feature for x-axis.
            conditioning_features (list): List of features to facet by.
        """
        data_df = pd.DataFrame(_dict)



        # 2. Plot conditional distributions
        for cond_feature in conditioning_features:
            if primary_feature == cond_feature:
                continue

            plt.figure()
            # Prepare data for faceted plot
            plot_df_cond = self.__prepare_data_for_dist_plotting(data_df.copy(), primary_feature, group_by=cond_feature)

            # Custom sort for age_range for primary feature x-axis
            if primary_feature == Const.age_range:
                x_order = sorted(plot_df_cond['Category'].unique(), key=lambda x: int(x.split('-')[0]) if '-' in x else x)
            else:
                x_order = None

            # Custom sort for age_range for conditioning feature columns
            if cond_feature == Const.age_range:
                col_order = sorted(data_df[cond_feature].unique(), key=lambda x: int(x.split('-')[0]) if '-' in x else x)
            else:
                col_order = None


            # Use catplot which is a figure-level interface for FacetGrid
            g = sns.catplot(
                data=plot_df_cond,
                kind='bar',
                x='Category',
                y='probability',
                hue='Distribution',
                col=cond_feature,
                col_wrap=4, # Wrap facets after 4 columns
                height=5,
                aspect=1.2,
                sharey=True, # Allow y-axis to differ between facets
                palette="mako",
                order=x_order, # Apply x-axis order
                col_order=col_order # Apply column order
            )

            g.fig.suptitle(f"Distribution of '{primary_feature.capitalize()}' by '{cond_feature.capitalize()}'", y=1.03)
            g.set_axis_labels(primary_feature.capitalize(), "Probability")
            g.set_titles("{col_name}") # Fixed f-string syntax
            g.tick_params(axis='x', which='both', rotation=45)
            g.tight_layout()
            plt.savefig(os.path.join(self.plot_dir,  f'{primary_feature}_{cond_feature}_comp_distribution_plot.svg'))
            print(f'created plot: {os.path.join(self.plot_dir, f'{primary_feature}_{cond_feature}_comp_distribution_plot.svg')}')
            plt.close('all')

    def _create_com_dist_plot_distance(self, data_df: pd.DataFrame, col: str, group_by: str):
        """
        Internal method to create comparative distribution plots with JS distance (w_dis).

        Args:
            data_df (pd.DataFrame): Data for plotting.
            col (str): Feature for columns/facets.
            group_by (str): Feature for x-axis.
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure()

        # Custom sort for age_range in FacetGrid columns
        if col == Const.age_range:
            col_order = sorted(data_df[col].unique(), key=lambda x: int(x.split('-')[0]) if '-' in x else x)
        else:
            col_order = None

        g = sns.FacetGrid(data_df, col=col, aspect=1, col_wrap=3 if len(data_df[col].unique()) >= 3 else len(data_df[col].unique()), height=6, col_order=col_order)

        # Use a list to store distances in order
        distances_list = []

        def draw_bar_plot_and_calculate_js(*args, **kwargs):
            """A helper function to draw a bar plot and calculate JS distance on a FacetGrid axis."""
            data = kwargs.pop('data')
            ax = kwargs.pop('ax')
            group_by_col = kwargs.pop('group_by_col')

            plot_data = self.__prepare_data_for_dist_plotting(data, group_by_col)

            # Custom sort for age_range on the x-axis within each facet
            if group_by_col == Const.age_range:
                x_order = sorted(plot_data['Category'].unique(), key=lambda x: int(x.split('-')[0]) if '-' in x else x)
            else:
                x_order = None


            # Calculate JS distance
            observed_counts = plot_data[plot_data['Distribution'] == 'Observed'].set_index('Category')['count']
            normative_counts = plot_data[plot_data['Distribution'] == 'Normative'].set_index('Category')['count']

             # Ensure both series have the same index for comparison
            all_categories = sorted(list(set(observed_counts.index) | set(normative_counts.index)))
            observed_counts = observed_counts.reindex(all_categories, fill_value=0)
            normative_counts = normative_counts.reindex(all_categories, fill_value=0)

            js_distance = jensenshannon(observed_counts.values, normative_counts.values) ** 2

            sns.barplot(data=plot_data,
                        x='Category',
                        y='probability',
                        hue='Distribution',
                        palette="mako",
                        ax=ax,
                        order=x_order) # Apply x-axis order

            return js_distance


        # Map the draw_heatmap function and collect returned distances
        for i, (name, group_data) in enumerate(g.facet_data()):
            ax = g.axes.flat[i]
            js_distance = draw_bar_plot_and_calculate_js(data=group_data, ax=ax, group_by_col=group_by)
            distances_list.append(js_distance)

        # Set the titles using the stored distances
        for i, ax in enumerate(g.axes.flat):
            if i < len(distances_list): # Ensure there's a distance for this axis
                title = ax.get_title()
                # Extract the group name from the existing title set by FacetGrid
                group_name = title.split(" | ")[-1].split("=")[-1] if " | " in title else title.split("=")[-1]
                ax.set_title(f"{group_name} - JS distance: {distances_list[i]:.4f}")

        g.tight_layout()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir,  f'{col}_{group_by}_comp_distribution_distance_plot.svg'))
        print(f'created plot: {os.path.join(self.plot_dir,  f'{col}_{group_by}_comp_distribution_distance_plot.svg')}')
        plt.close('all')

    def create_comp_distribution_plots_w_dis(self, _dict: dict, primary_feature: str, conditioning_features: list):

        data_df = pd.DataFrame(_dict)

        for cond_feature in conditioning_features:
            if primary_feature == cond_feature:
                continue

            self._create_com_dist_plot_distance(data_df=data_df, col=primary_feature, group_by=cond_feature)