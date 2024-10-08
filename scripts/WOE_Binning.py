import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

def woe_binning(data, target, n_bins=10):
    """
    Perform WoE binning for Recency, Frequency, and Monetary.
    :param data: DataFrame containing features and the target variable.
    :param target: Target variable (Good/Bad label).
    :param n_bins: Number of bins for discretization.
    :return: DataFrame with WoE values for each feature.
    """
    woe_df = pd.DataFrame()
    
    for feature in ['Recency', 'Frequency', 'Monetary']:
        # Discretize features into bins
        binner = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        data[f'{feature}_bin'] = binner.fit_transform(data[[feature]])

        # Group by bins and calculate Good/Bad counts
        bin_stats = data.groupby(f'{feature}_bin').agg(
            Total=('Risk_Label', 'size'),
            Bad=('Risk_Label', 'sum'),
            Good=('Risk_Label', lambda x: (x == 0).sum())
        )
        
        # Avoid division by zero
        bin_stats['Good'] = bin_stats['Good'].replace(0, 0.5)
        bin_stats['Bad'] = bin_stats['Bad'].replace(0, 0.5)
        
        # Calculate WoE for each bin
        bin_stats[f'WoE_{feature}'] = np.log(
            (bin_stats['Good'] / bin_stats['Good'].sum()) / 
            (bin_stats['Bad'] / bin_stats['Bad'].sum())
        )
        
        # Add WoE to final dataframe
        woe_df = pd.concat([woe_df, bin_stats[f'WoE_{feature}']], axis=1)
    
    return woe_df
