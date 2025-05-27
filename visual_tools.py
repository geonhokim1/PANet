import matplotlib.pyplot as plt

def plot_graph(train_data, val_data, epoch, phase):
    plt.figure(figsize=(12,8))
    plt.plot(train_data, label='Train', color='blue')
    plt.plot(val_data, label='Validation', color='red')
    plt.xlabel("Epoch")
    plt.ylabel(f"{phase}")
    plt.title(f"{phase} Graph")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./plot_graph/{phase}_graph_epoch{epoch}.png", dpi=300)
    plt.close()