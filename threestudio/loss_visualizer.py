import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def merge_alternating_lines(df):
    # We assume that the rows with NaNs are to be merged with the next row
    merged_rows = []
    for i in range(0, len(df), 2):
        if i+1 < len(df):
            merged_row = df.iloc[i].combine_first(df.iloc[i+1])
            merged_rows.append(merged_row)
    merged_df = pd.DataFrame(merged_rows)
    return merged_df

def sanitize_folder_name(folder_name):
    # Replace slashes and other special characters with underscores
    sanitized_name = folder_name.replace(os.sep, '_').replace(':', '_').replace(' ', '_')
    return sanitized_name

def plot_and_save_loss_columns(df, save_dir, filename, cache=True):
    loss_columns = [col for col in df.columns if 'loss' in col.lower()]
    for col in loss_columns:
        save_path = os.path.join(save_dir, f'{filename}_{col}.png')
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if cache and os.path.exists(save_path):
            continue  # Skip if the plot already exists and caching is enabled

        plt.figure(figsize=(10, 6))
        plt.plot(df['step'], df[col], label=col)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title(f'{col} vs Step in {filename}')
        plt.legend()
        plt.grid(True)

        # Calculate and display summary statistics
        mean_val = df[col].mean()
        std_val = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()
        summary_text = f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMin: {min_val:.4f}\nMax: {max_val:.4f}'
        plt.gcf().text(0.15, 0.75, summary_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

        # Save the plot
        plt.savefig(save_path)
        plt.close()

def process_csv_files(root_directory, cache=True):
    # Collect all the metrics.csv file paths
    file_paths = []
    for root, dirs, files in os.walk(root_directory):
        if 'csv_logs/version_0' in root:
            file_path = os.path.join(root, 'metrics.csv')
            if os.path.exists(file_path):
                file_paths.append((file_path, root))

    # Process each file and visualize the progress
    for file_path, folder_name in tqdm(file_paths, desc="Processing CSV files"):
        data = pd.read_csv(file_path)
        merged_data = merge_alternating_lines(data)
        breakpoint()
        plot_and_save_loss_columns(merged_data, save_dir="/".join(folder_name.split("/")[:-2]), filename=folder_name.split("/")[-3], cache=cache)





##### FOR PROCESSING MULTIPLE FOLDERS AT ONCE
# Set the root directory path
root_directory = '/media/hdd/donghoon/3DConst/outputs/gau_stable_diffusion'
# Process all CSV files and plot the loss columns
process_csv_files(root_directory, cache=False) # cache: if you specify cache=True, it will not plot the result if there is already plot png file.





##### FOR SINGLE FOLDER TEST!!!
# file_path = '/media/hdd/donghoon/3DConst/outputs/gau_stable_diffusion/a_goose_made_out_of_gold_0805-1-blur_cause_ablation@baseline@20240805-052834/csv_logs/version_0/metrics.csv'
# folder_name = '/media/hdd/donghoon/3DConst/outputs/gau_stable_diffusion/a_goose_made_out_of_gold_0805-1-blur_cause_ablation@baseline@20240805-052834/csv_logs/version_0'
# data = pd.read_csv(file_path)
# breakpoint()
# merged_data = merge_alternating_lines(data)
# plot_and_save_loss_columns(merged_data, folder_name, os.path.dirname(file_path), cache=False)

