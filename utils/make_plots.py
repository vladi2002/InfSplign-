import matplotlib.pyplot as plt

# warm start 1 update after
losses = [2.2649765014648438e-06, 3.0994415283203125e-06, 6.139278411865234e-06, 7.271766662597656e-06, 1.1444091796875e-05, 1.5497207641601562e-05, 2.1576881408691406e-05, 3.170967102050781e-05, 2.396106719970703e-05, 2.4437904357910156e-05, 4.774332046508789e-05, 6.186962127685547e-05, 5.060434341430664e-05, 9.28044319152832e-05, 0.00010633468627929688, 9.894371032714844e-05, 8.600950241088867e-05, 8.034706115722656e-05, 6.818771362304688e-05, 9.28640365600586e-05, 8.314847946166992e-05, 6.258487701416016e-05, 6.908178329467773e-05, 7.12275505065918e-05, 5.9545040130615234e-05, 5.030632019042969e-05]
gradients = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (-5.364418029785156e-07, 4.76837158203125e-07), (-5.364418029785156e-07, 3.5762786865234375e-07), (-7.748603820800781e-07, 1.3709068298339844e-06), (-5.364418029785156e-07, 5.364418029785156e-07), (-4.172325134277344e-07, 4.76837158203125e-07), (-4.76837158203125e-07, 5.364418029785156e-07), (-2.086162567138672e-06, 5.364418029785156e-06), (-4.231929779052734e-06, 4.231929779052734e-06), (-3.4570693969726562e-06, 4.291534423828125e-06), (-3.635883331298828e-06, 6.258487701416016e-06), (-5.9604644775390625e-06, 7.450580596923828e-06), (-4.172325134277344e-06, 5.125999450683594e-06), (-4.76837158203125e-06, 4.76837158203125e-06), (-3.159046173095703e-06, 5.185604095458984e-06), (-4.291534423828125e-06, 4.291534423828125e-06), (-5.125999450683594e-06, 5.245208740234375e-06), (-4.351139068603516e-06, 4.172325134277344e-06), (-3.635883331298828e-06, 4.231929779052734e-06), (-4.708766937255859e-06, 3.933906555175781e-06), (-4.351139068603516e-06, 5.245208740234375e-06), (-4.410743713378906e-06, 3.814697265625e-06), (-3.4570693969726562e-06, 4.470348358154297e-06)]
latents = [(-3.978515625, 3.9453125), (-4.0234375, 4.1015625), (-3.59765625, 4.3203125), (-3.642578125, 4.58984375), (-3.9921875, 4.3046875), (-3.9921875, 4.1171875), (-3.978515625, 3.75), (-4.30078125, 4.359375), (-3.75, 4.1171875), (-3.59375, 3.82421875), (-4.46484375, 4.2578125), (-4.37109375, 3.857421875), (-4.34765625, 3.576171875), (-4.6796875, 3.578125), (-4.234375, 3.666015625), (-3.708984375, 3.599609375), (-3.96484375, 3.86328125), (-3.6640625, 3.7421875), (-3.48046875, 4.0625), (-3.59375, 4.37890625), (-3.68359375, 4.6171875), (-3.75, 4.41015625), (-3.73828125, 4.125), (-3.611328125, 4.44140625), (-3.60546875, 4.9296875), (-3.71484375, 4.82421875)]


def plot_everything(losses, gradients, latents):
    timesteps = list(range(50, 24, -1))

    # Plot for losses
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, losses, label='Loss', marker='o')
    plt.xlabel('Timestep')
    plt.ylabel('Loss Value')
    plt.title('Loss vs Timestep')
    plt.gca().invert_xaxis()  # Invert x-axis to match timestep order
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot for gradients
    min_gradients = [grad[0] for grad in gradients]
    max_gradients = [grad[1] for grad in gradients]

    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, min_gradients, label='Min Gradient', color='blue', marker='o')
    plt.plot(timesteps, max_gradients, label='Max Gradient', color='red', marker='o')
    plt.xlabel('Timestep')
    plt.ylabel('Gradient Value')
    plt.title('Gradient Min/Max vs Timestep')
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot for latents
    min_latents = [lat[0] for lat in latents]
    max_latents = [lat[1] for lat in latents]

    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, min_latents, label='Min Latent', color='blue', marker='o')
    plt.plot(timesteps, max_latents, label='Max Latent', color='red', marker='o')
    plt.xlabel('Timestep')
    plt.ylabel('Latent Value')
    plt.title('Latent Min/Max vs Timestep')
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_everything(losses, gradients, latents)
