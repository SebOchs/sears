import matplotlib.pyplot as plt
import numpy as np






def plot_img(file_loc):
    file = np.load(file_loc, allow_pickle=True).item()
    plot = sorted(file.items(), key=lambda item: item[1], reverse=True)
    rng = np.arange(10)
    plt.rcdefaults()
    fig, ax = plt.subplots()
    ax.barh(rng, [x[1] for x in plot][:10])
    ax.set_yticks(rng)
    ax.set_yticklabels([x[0] for x in plot][:10])
    ax.set_xlabel("Adversary Count")
    ax.set_title("Top 10 Performing Adverbs")
    plt.tight_layout()
    plt.savefig(file_loc[:-4] + '.png', dpi=300)


def main():
    plot_img('adj_result.npy')
    #plot_img('adv_result.npy')


if __name__ == "__main__":
    main()