import argparse
import json
import os
import matplotlib.pyplot as plt


def plot_training_from_json(json_path, output_dir=None, exp_name=None):
    """Generate training plots from a metrics.json file"""
    
    # Load metrics
    with open(json_path, 'r') as f:
        metrics = json.load(f)
    
    if not exp_name:
        exp_name = os.path.basename(os.path.dirname(json_path))
    
    if output_dir is None:
        output_dir = os.path.dirname(json_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Training Metrics - {exp_name}', fontsize=16)
    
    # Plot 1: Loss curve
    ax = axes[0, 0]
    if metrics.get('iterations') and metrics.get('loss_pixel'):
        ax.plot(metrics['iterations'], metrics['loss_pixel'], 'b-', linewidth=2, label='Pixel Loss')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No loss data available', ha='center', va='center', transform=ax.transAxes)
    
    # Plot 2: Learning Rate curve
    ax = axes[0, 1]
    if metrics.get('iterations') and metrics.get('learning_rate'):
        ax.plot(metrics['iterations'], metrics['learning_rate'], 'g-', linewidth=2, label='Learning Rate')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('LR')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_yscale('log')
    else:
        ax.text(0.5, 0.5, 'No LR data available', ha='center', va='center', transform=ax.transAxes)
    
    # Plot 3: PSNR curves for all datasets
    ax = axes[1, 0]
    colors = ['r', 'm', 'c', 'y', 'k']
    has_psnr = False
    
    if metrics.get('psnr_valid'):
        for idx, (dataset, psnr_data) in enumerate(metrics['psnr_valid'].items()):
            if psnr_data:
                iters = [p['iter'] for p in psnr_data]
                psnrs = [p['psnr'] for p in psnr_data]
                color = colors[idx % len(colors)]
                ax.plot(iters, psnrs, f'{color}o-', linewidth=2, markersize=4, label=f'{dataset} PSNR')
                has_psnr = True
    
    if has_psnr:
        ax.set_xlabel('Iteration')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('Validation PSNR')
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No PSNR data available', ha='center', va='center', transform=ax.transAxes)
    
    # Plot 4: Combined Loss and PSNR (dual axis)
    ax = axes[1, 1]
    if metrics.get('iterations') and metrics.get('loss_pixel'):
        ax.plot(metrics['iterations'], metrics['loss_pixel'], 'b-', linewidth=2, label='Pixel Loss')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax.grid(True, alpha=0.3)
        
        ax2 = ax.twinx()
        has_psnr_dual = False
        
        if metrics.get('psnr_valid'):
            for idx, (dataset, psnr_data) in enumerate(metrics['psnr_valid'].items()):
                if psnr_data:
                    iters = [p['iter'] for p in psnr_data]
                    psnrs = [p['psnr'] for p in psnr_data]
                    color = colors[idx % len(colors)]
                    ax2.plot(iters, psnrs, f'{color}s--', linewidth=2, markersize=4, label=f'{dataset} PSNR')
                    has_psnr_dual = True
        
        if has_psnr_dual:
            ax2.set_ylabel('PSNR (dB)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.set_title('Loss vs PSNR')
            
            # Combined legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax.set_title('Training Loss')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_plots.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training plots saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser("Plot Training Metrics from JSON")
    parser.add_argument("--json-path", type=str, required=True, 
                        help="Path to metrics.json file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for plots (default: same as JSON)")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Experiment name for title (default: directory name)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json_path):
        print(f"Error: JSON file not found: {args.json_path}")
        return
    
    plot_training_from_json(args.json_path, args.output_dir, args.exp_name)


if __name__ == "__main__":
    main()
