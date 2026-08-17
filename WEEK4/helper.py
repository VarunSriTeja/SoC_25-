import matplotlib.pyplot as plt
import os

plt.ion()

def plot(scores, mean_scores, mode=None, n_games=None):
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(round(mean_scores[-1], 2)))
    plt.legend()
    plt.tight_layout()
    plt.pause(0.1) 

    if mode is not None and n_games is not None:
        folder = "plots"
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = f"{mode}_{n_games}games.png"
        filepath = os.path.join(folder, filename)
        plt.savefig(filepath)
        print(f"✅ Plot saved: {filepath}")
