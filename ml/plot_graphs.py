# plot_graphs.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import config as c
from scipy import stats

def generate_plots():
    if not os.path.exists(c.SUMMARY_LOG_PATH):
        print(f"Error: Data file {c.SUMMARY_LOG_PATH} not found. Run simulation first.")
        return

    print(f"Reading data from {c.SUMMARY_LOG_PATH}...")
    try:
        df = pd.read_csv(c.SUMMARY_LOG_PATH)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Check if data exists
    if df.empty:
        print("CSV is empty. Simulation might have failed or is still running.")
        return
    
    # Rename models for consistency
    df['Model'] = df['Model'].replace({
        'BASELINE': 'Without EE-AURS',
        'SVM': 'SVM with EE-AURS',
        'RF': 'RF with EE-AURS',
        'DTC': 'DTC with EE-AURS'
    })
    
    # Get unique models and node counts
    desired_order = ['Without EE-AURS', 'SVM with EE-AURS', 'RF with EE-AURS', 'DTC with EE-AURS']
    existing_models = df['Model'].unique()
    models = [m for m in desired_order if m in existing_models]
    
    node_counts = sorted(df['Nodes'].unique())
    
    # Set up plot styles with distinct colors
    colors = {
        'Without EE-AURS': '#D62728',      # Red
        'SVM with EE-AURS': '#1F77B4',     # Blue
        'RF with EE-AURS': '#2CA02C',      # Green
        'DTC with EE-AURS': '#FF7F0E'      # Orange/Gold
    }
    
    markers = {
        'Without EE-AURS': 'v',
        'SVM with EE-AURS': 'o',
        'RF with EE-AURS': 's',
        'DTC with EE-AURS': 'x'
    }
    
    line_styles = {
        'Without EE-AURS': '--',
        'SVM with EE-AURS': '-',
        'RF with EE-AURS': '-.',
        'DTC with EE-AURS': ':'
    }

    # Create Graphs directory if not exists
    if not os.path.exists(c.GRAPH_DIR):
        os.makedirs(c.GRAPH_DIR)

    metrics = ['PDR', 'RoR', 'ECR', 'E2ED']
    ylabels = {
        'PDR': 'Packet Delivery Ratio (%)',
        'RoR': 'Routing Overhead Ratio (%)',
        'ECR': 'Energy Consumption Ratio (%)',
        'E2ED': 'End-to-End Delay (s)'
    }
    
    # Process data for each model and metric
    processed_data = {}
    
    for model in models:
        processed_data[model] = {}
        for metric in metrics:
            processed_data[model][metric] = {'means': [], 'nodes': []}
            
            for nodes in node_counts:
                values = df[(df['Model'] == model) & (df['Nodes'] == nodes)][metric].values
                
                if len(values) > 0:
                    if metric in ['PDR', 'RoR', 'ECR']:
                        values = values * 100
                    
                    mean_val = np.mean(values)
                    
                    processed_data[model][metric]['means'].append(mean_val)
                    processed_data[model][metric]['nodes'].append(nodes)
    
    print(f"Generating combined comparison graphs (No Error Bars)...")
    
    # =================================================================
    # GENERATE COMBINED GRAPHS (Clean Lines)
    # =================================================================
    
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        y_min_global = float('inf')
        y_max_global = float('-inf')
        
        for model in models:
            # Skip Baseline for ECR graph only
            if metric == 'ECR' and model == 'Without EE-AURS':
                continue

            if model in processed_data and metric in processed_data[model]:
                data = processed_data[model][metric]
                
                if len(data['means']) > 0:
                    x_vals = np.array(data['nodes'])
                    y_vals = np.array(data['means'])
                    
                    y_min_global = min(y_min_global, np.min(y_vals))
                    y_max_global = max(y_max_global, np.max(y_vals))
                    
                    # [CHANGE] Use plot instead of errorbar
                    ax.plot(x_vals, y_vals,
                            label=model,
                            color=colors.get(model, 'black'),
                            marker=markers.get(model, 'o'),
                            linestyle=line_styles.get(model, '-'),
                            linewidth=2.5,
                            markersize=9)
        
        # --- DYNAMIC Y-AXIS SCALING ---
        if y_max_global > float('-inf'):
            y_range = y_max_global - y_min_global
            
            if y_range == 0:
                padding = y_max_global * 0.1 if y_max_global != 0 else 1.0
            else:
                padding = y_range * 0.15 
            
            y_bottom = max(0, y_min_global - padding)
            y_top = y_max_global + padding
            
            if metric in ['PDR', 'RoR', 'ECR']:
                y_top = min(105, y_top) 
                
            ax.set_ylim(y_bottom, y_top)

        ax.set_xlabel("Number of nodes", fontweight='bold', fontsize=14)
        ax.set_ylabel(ylabels[metric], fontweight='bold', fontsize=14)
        ax.set_xticks(node_counts)
        ax.set_xticklabels([f"{n}" for n in node_counts])
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, framealpha=0.9, loc='best')
        
        if metric in ['PDR', 'RoR', 'ECR']:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        else:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        
        ax.tick_params(labelsize=12)
        ax.set_title(f"{ylabels[metric].split('(')[0].strip()}", fontsize=16, fontweight='bold', pad=15)
        
        plt.tight_layout()
        
        filename_base = f"Combined_{metric}"
        png_path = os.path.join(c.GRAPH_DIR, f"{filename_base}.png")
        pdf_path = os.path.join(c.GRAPH_DIR, f"{filename_base}.pdf")
        
        plt.savefig(png_path, dpi=600, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        
        print(f"  Saved: {png_path}")
        plt.close(fig)
    
    print("\nGraph Generation Complete!")

if __name__ == "__main__":
    generate_plots()