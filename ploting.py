import matplotlib.pyplot as plt
import os

def read_losses(filename):
    steps = []
    losses = []
    step=0
    counter=0
    with open(filename, 'r') as f:
        for line in f:
            if line.strip()and counter%9==0:  # skip empty lines
                loss = float(line.strip())
                steps.append(int(step))
                step+=1
                losses.append(loss)
            counter+=1
    return steps, losses

def plot_losses(name):
    # Read data from both files
    steps1, losses1 = read_losses('obj1.txt')
    steps2, losses2 = read_losses('obj2.txt')
    steps3, losses_total = read_losses('objs.txt')
    steps4, spatial = read_losses('spatial.txt')

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(steps1, losses1, label='Object 1')
    plt.plot(steps2, losses2, label='Object 2')
    plt.plot(steps3, losses_total, label='Total Loss')
    plt.plot(steps4, spatial, label='Spatial')


    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss Comparison_'+name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    # Save the figure
    plt.savefig('plots/loss_plot_'+name+".png")
    os.system('rm obj1.txt')
    os.system('rm obj2.txt')
    os.system('rm objs.txt')
    os.system('rm spatial.txt')

    plt.show()