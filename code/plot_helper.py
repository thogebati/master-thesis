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

    gen_dir = '../results/plots'
    plot_dir = ''

    def __init__(self, date: str):
        self.plot_dir = os.path.join(self.gen_dir, date)
        os.makedirs(self.plot_dir, exist_ok=True)

    def __prepare_data_for_dist_plotting(self, df, feature, group_by=None):
        """
        Prepares data in a long-form DataFrame suitable for seaborn plotting.
        Calculates both observed and normative (uniform) distributions.
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

    def create_race_gender_mosaic_plot(self, cobined_dict:dict):
        """
        Creates a mosaic plot to visualize intersectional bias between race and gender.
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
            
    def _create_heatmap(self, data_df:pd.DataFrame, values:str, index:str, columns:str, filename:str, title:str, xlabel:str, ylabel:str):
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
        
    def _create_boxplots(self, data_df:pd.DataFrame, xvalues:str, yvalues:str, filename:str, title:str, xlabel:str, ylabel:str):
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure()
        
        sns.boxplot(data=data_df, x=xvalues, y=yvalues, palette="mako", orient="v")

        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.xticks(rotation=45, ha='right')
        # plt.xticks(rotation=45, ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, filename))
        print(f'created plot: {os.path.join(self.plot_dir, filename)}')
        plt.close('all')

    def create_boxplot(self, _gen_dict:dict, x_type:str, y_type:str):
        df = pd.DataFrame(_gen_dict)
        scaler = MinMaxScaler()
        df['mean_depth'] = scaler.fit_transform(df[['mean_depth']])
        self._create_boxplots(data_df=df, xvalues=x_type, yvalues=y_type, ylabel=y_type, filename=f'{x_type}_{y_type}_boxplot.svg', title=f'{y_type}s distribution {x_type}', xlabel=x_type)
        
    def create_heatmap(self, _gen_dict: dict, values: str, index: str, columns: str):
        df = pd.DataFrame(_gen_dict)
        self._create_heatmap(data_df=df, values=values, index=index, columns=columns, ylabel=index, filename=f'{index}_{columns}_{values}_heatmap.svg', title=f'average {values} of {columns} and {index} combined', xlabel=columns)

    def create_multidimensional_heatmap(self, _gen_dict: dict, values: str, index: str, columns: str, grouping: str, low_count_threshold: int = 20):
        data_df = pd.DataFrame(_gen_dict)

        # Define the order of categories for consistent plotting
        index_order = [r for r in Const.possible_values[index] if r in data_df[index].unique()]
        columns_order = [g for g in Const.possible_values[columns] if g in data_df[columns].unique()]
        
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
        g = sns.FacetGrid(data_df, col=grouping, aspect=1, col_wrap=3, height=6,)
        
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
        data_df = pd.DataFrame(_gen_dict)

        # Define the order of categories for consistent plotting
        index_order = [r for r in Const.possible_values[index] if r in data_df[index].unique()]
        columns_order = [g for g in Const.possible_values[columns] if g in data_df[columns].unique()]
        
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
        g = sns.FacetGrid(data_df, col=major_grouping, row=grouping, aspect=1, height=6)
        
        # Map the heatmap drawing function to each facet in the grid.
        # We pass vmin and vmax to ensure a consistent color scale across all plots.
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

    def __create_simpl_distribution_plot(self, observed_dist, normative_dist, labels, title, filename):
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
        types = [face[type] for face in _dict]
        type_counts = Counter(types)
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
        
    def create_comp_distribution_plots_wout_dis(self, _dict: dict, primary_feature, conditioning_features):
        """
        Creates a set of FacetGrid plots for a primary feature.
        1. Overall distribution vs. Normative.
        2. Conditional distributions vs. Normative, faceted by other features.
        """
        data_df = pd.DataFrame(_dict)

        

        # 2. Plot conditional distributions
        for cond_feature in conditioning_features:
            if primary_feature == cond_feature:
                continue
            
            plt.figure()
            # Prepare data for faceted plot
            plot_df_cond = self.__prepare_data_for_dist_plotting(data_df.copy(), primary_feature, group_by=cond_feature)

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
                palette="mako"
            )

            g.fig.suptitle(f"Distribution of '{primary_feature.capitalize()}' by '{cond_feature.capitalize()}'", y=1.03)
            g.set_axis_labels(primary_feature.capitalize(), "Probability")
            g.set_titles("{col_name}") # Fixed f-string syntax
            g.tick_params(axis='x', which='both', rotation=45)
            g.tight_layout()
            plt.savefig(os.path.join(self.plot_dir,  f'{primary_feature}_{cond_feature}_comp_distribution_plot.svg'))
            print(f'created plot: {os.path.join(self.plot_dir, f'{primary_feature}_{cond_feature}_comp_distribution_plot.svg')}')
            plt.close('all')  
            
    def _create_com_dist_plot_distance(self, data_df: pd.DataFrame, col:str, group_by: str):
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure()
        
        def draw_box_plot(*args, **kwargs):
            """A helper function to draw a heatmap on a FacetGrid axis."""
            # This function receives a subset of the data for each facet
            data = kwargs.pop('data')
            # Get the current axis
            ax = kwargs.pop('ax')

            data = self.__prepare_data_for_dist_plotting(data, kwargs.pop('group_by'))
            grouped = data.groupby('Distribution')

            # Get observed and normative counts
            normative_counts = grouped.get_group('Normative')['count'].values
            num_ = len(normative_counts)
            normative_dist = [1/num_] * num_

            # Calculate Jensen-Shannon distance
            js_distance = jensenshannon(normative_counts, normative_dist) ** 2

            # You can choose to print or store the js_distance, for now let's print it
            # print(f"Jensen-Shannon distance: {js_distance}") # Removed printing here

            sns.barplot(data,
                x='Category',
                y='probability',
                hue='Distribution',
                palette="mako",
                ax=ax,) # Pass the current axis to seaborn

            # Return the calculated distance
            return js_distance
        
        groups = data_df.groupby(col).groups
        
        g = sns.FacetGrid(data_df, col=col, aspect=1, col_wrap=3 if len(groups) >= 3 else len(groups), height=6)

        # Use a list to store distances in order
        distances_list = []

        # Map the draw_heatmap function and collect returned distances
        for i, (name, group_data) in enumerate(g.facet_data()):
            ax = g.axes.flat[i]
            js_distance = draw_box_plot(data=group_data, ax=ax, group_by=group_by)
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
            
    def create_comp_distribution_plots_w_dis(self, _dict: dict, primary_feature, conditioning_features):
        
        data_df = pd.DataFrame(_dict)

        for cond_feature in conditioning_features:
            if primary_feature == cond_feature:
                continue
            
            self._create_com_dist_plot_distance(data_df=data_df, col=primary_feature, group_by=cond_feature)