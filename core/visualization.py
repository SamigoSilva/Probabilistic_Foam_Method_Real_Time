import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_comparison(results: dict):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, metric in zip(axs, ['coverage', 'entropy', 'uniformity']):
        for method, values in results.items():
            ax.plot([v['metrics'][metric] for v in values], label=method)
        ax.set_title(metric)
        ax.legend()
    
    plt.savefig('comparison.png')

def animate_evolution(foam_history: list):
    fig = plt.figure()
    im = plt.imshow(foam_history[0], animated=True)
    
    def update(frame):
        im.set_array(foam_history[frame])
        return im,
    
    anim = FuncAnimation(fig, update, frames=len(foam_history), blit=True)
    anim.save('evolution.mp4', writer='ffmpeg')