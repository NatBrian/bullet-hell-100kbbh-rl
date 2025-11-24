import os
import pandas as pd
import matplotlib.pyplot as plt

def save_plots(log_dir):
    """Generates and saves reward and loss plots from training log."""
    csv_path = os.path.join(log_dir, "training_log.csv")
    if not os.path.exists(csv_path):
        print("No training log found.")
        return

    df = pd.read_csv(csv_path)
    
    # Plot Reward
    plt.figure(figsize=(10, 5))
    plt.plot(df['episode'], df['reward'], label='Reward')
    if len(df) > 10:
        plt.plot(df['episode'], df['reward'].rolling(window=10).mean(), label='Avg Reward (10)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.title('Training Reward')
    plt.savefig(os.path.join(log_dir, "reward_plot.png"))
    plt.close()

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(df['episode'], df['loss'], label='Loss', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(os.path.join(log_dir, "loss_plot.png"))
    plt.close()

def generate_html_report(log_dir, output_file="report.html"):
    """Creates a standalone HTML report."""
    csv_path = os.path.join(log_dir, "training_log.csv")
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    
    # Ensure plots exist
    save_plots(log_dir)
    
    html = f"""
    <html>
    <head>
        <title>RL Training Report</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; }}
            img {{ border: 1px solid #ddd; margin: 10px; }}
            table {{ border-collapse: collapse; width: 50%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>100KBBH RL Training Report</h1>
        <h2>Summary Statistics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Episodes</td><td>{len(df)}</td></tr>
            <tr><td>Max Reward</td><td>{df['reward'].max()}</td></tr>
            <tr><td>Average Reward (Last 100)</td><td>{df['reward'].tail(100).mean():.2f}</td></tr>
            <tr><td>Total Steps</td><td>{df['steps'].sum()}</td></tr>
        </table>
        
        <h2>Training Curves</h2>
        <div>
            <img src="reward_plot.png" width="600" alt="Reward Plot">
            <img src="loss_plot.png" width="600" alt="Loss Plot">
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(log_dir, output_file), "w") as f:
        f.write(html)
    print(f"Report generated at {os.path.join(log_dir, output_file)}")
