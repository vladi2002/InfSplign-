import fastcore.all as fc
from PIL import Image
import math, random, torch, matplotlib.pyplot as plt, numpy as np, matplotlib as mpl
from itertools import zip_longest
from diffusers.utils import PIL_INTERPOLATION
from math import sqrt
import os
from sklearn.decomposition import PCA
import torchvision.transforms as T
# from textwrap import wrap
import matplotlib

@fc.delegates(plt.Axes.imshow)
def show_image(im, ax=None, figsize=None, title=None, noframe=True, save_orig=False,**kwargs):
    "Show a PIL or PyTorch image on `ax`."
    if save_orig: im.save('orig.png')
    if fc.hasattrs(im, ('cpu','permute','detach')):
        im = im.detach().cpu()
        if len(im.shape)==3 and im.shape[0]<5: im=im.permute(1,2,0)
    elif not isinstance(im,np.ndarray): im=np.array(im)
    if im.shape[-1]==1: im=im[...,0]
    if ax is None: _,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, **kwargs)
    if title is not None: ax.set_title(title)
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    if noframe: ax.axis('off')
    return ax

@fc.delegates(plt.subplots, keep=True)
def subplots(
    nrows:int=1, # Number of rows in returned axes grid
    ncols:int=1, # Number of columns in returned axes grid
    figsize:tuple=None, # Width, height in inches of the returned figure
    imsize:int=3, # Size (in inches) of images that will be displayed in the returned figure
    suptitle:str=None, # Title to be set to returned figure
    **kwargs
): # fig and axs
    "A figure and set of subplots to display images of `imsize` inches"
    if figsize is None: figsize=(ncols*imsize, nrows*imsize)
    fig,ax = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    if suptitle is not None: fig.suptitle(suptitle)
    if nrows*ncols==1: ax = np.array([ax])
    return fig,ax

@fc.delegates(subplots)
def get_grid(
    n:int, # Number of axes
    nrows:int=None, # Number of rows, defaulting to `int(math.sqrt(n))`
    ncols:int=None, # Number of columns, defaulting to `ceil(n/rows)`
    title:str=None, # If passed, title set to the figure
    weight:str='bold', # Title font weight
    size:int=14, # Title font size
    **kwargs,
): # fig and axs
    "Return a grid of `n` axes, `rows` by `cols`"
    if nrows: ncols = ncols or int(np.floor(n/nrows))
    elif ncols: nrows = nrows or int(np.ceil(n/ncols))
    else:
        nrows = int(math.sqrt(n))
        ncols = int(np.floor(n/nrows))
    fig,axs = subplots(nrows, ncols, **kwargs)
    for i in range(n, nrows*ncols): axs.flat[i].set_axis_off()
    if title is not None: fig.suptitle(title, weight=weight, size=size)
    return fig,axs

@fc.delegates(subplots)
def show_images(ims:list, # Images to show
                nrows:int=None, # Number of rows in grid
                ncols:int=None, # Number of columns in grid (auto-calculated if None)
                titles:list=None, # Optional list of titles for each image
                save_orig:bool=False, # If True, save original image
                **kwargs):
    "Show all images `ims` as subplots with `rows` using `titles`"
    axs = get_grid(len(ims), nrows, ncols, **kwargs)[1].flat
    for im,t,ax in zip_longest(ims, titles or [], axs): show_image(im, ax=ax, title=t, save_orig=save_orig)

def preprocess_image(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def visualize_and_save_features_pca(feature_maps_fit_data,feature_maps_transform_data, transform_experiments, t):
    feature_maps_fit_data = feature_maps_fit_data.cpu().numpy()
    pca = PCA(n_components=3)
    pca.fit(feature_maps_fit_data)
    feature_maps_pca = pca.transform(feature_maps_transform_data.cpu().numpy())  # N X 3
    feature_maps_pca = feature_maps_pca.reshape(len(transform_experiments), -1, 3)  # B x (H * W) x 3
    for i, experiment in enumerate(transform_experiments):
        pca_img = feature_maps_pca[i]  # (H * W) x 3
        h = w = int(sqrt(pca_img.shape[0]))
        pca_img = pca_img.reshape(h, w, 3)
        pca_img_min = pca_img.min(axis=(0, 1))
        pca_img_max = pca_img.max(axis=(0, 1))
        pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
        pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
        pca_img = T.Resize(512, interpolation=T.InterpolationMode.NEAREST)(pca_img)
        pca_img.save(os.path.join(f"{experiment}_time_{t}.png"))


def prepare_attention_map(shape):
    # shape = shape.mean(dim=0).squeeze()
    shape_np = shape.cpu().detach().numpy()

    if shape_np.ndim == 1 or (shape_np.ndim == 2 and shape_np.shape[0] == 1):
        N = shape_np.size
        h = w = int(np.sqrt(N))
        shape_img = shape_np.reshape(h, w)
    elif shape_np.ndim == 2:
        shape_img = shape_np
    else:
        shape_img = shape_np.squeeze()

    return shape_img


def show_attention_maps(orig, target, block, timestep, counter, word):
    orig_img = prepare_attention_map(orig)
    move_to_img = prepare_attention_map(target)

    save_dir = "attention_maps"
    os.makedirs(save_dir, exist_ok=True)

    title_orig = "orig"
    title_move = f"target t={timestep} {block}_block {target.shape}"
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(orig_img, cmap="viridis")
    axs[0].set_title(title_orig)
    axs[0].axis("off")
    axs[1].imshow(move_to_img, cmap="viridis")
    axs[1].set_title(title_move)
    axs[1].axis("off")
    # plt.show()
    # fig.savefig(os.path.join(save_dir, f"target_{word}_timestep_{timestep}_{block}_block_{target.shape}_{counter}.png"))


def get_attention_map(edit, timestep, block, word, prompt, seed=0, sample=0, counter=None):
    edit_img = prepare_attention_map(edit)

    save_dir = f"attn_maps_{seed}"
    save_dir = os.path.join("results", prompt, save_dir)
    # print(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(edit_img, cmap="viridis")
    # edit_title = f"edit {word} t={timestep} {block}_block {edit.shape}"
    edit_title = f"{word} t={timestep} {block}"
    ax.set_title(edit_title)
    ax.axis("off")
    # plt.show()
    # fig.savefig(os.path.join(save_dir, f"edit_{word}_timestep_{timestep}_{block}_block_{edit.shape}_{counter}.png"))
    os.makedirs(os.path.join("clusters", f"{prompt}_{sample}"), exist_ok=True)
    fig.savefig(os.path.join("clusters", f"{prompt}_{sample}", f"t={timestep} word={word}.png"))
    # fig.savefig(os.path.join(save_dir, f"cat loss t={timestep} word={word}.png"))


def plot_attention_map(edit, timestep, block, centroid=None, object="",
                       loss_type="relu", loss_num=1, prompt="",
                       margin=0.5, alpha=1, attn_folder="attention_maps", img_id=""):
    edit_img = prepare_attention_map(edit)

    # save_dir = os.path.join(loss_type, "attention_maps")
    save_dir = os.path.join("attention_maps", img_id, f"{prompt}_sd1.4_{loss_type}",f"{object}",f"{block}")
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(edit_img, cmap="viridis")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Value')

    if centroid is not None:
        centroid = centroid.squeeze()  # should now be shape [2]
        x_norm, y_norm = centroid.tolist()  # normalized coordinates in [0, 1]

        # Convert normalized coordinates to pixel coordinates
        height, width = edit_img.shape[0], edit_img.shape[1]
        x_pixel = x_norm * width
        y_pixel = y_norm * height

        ax.scatter(x_pixel, y_pixel, s=300, c='red', marker='x', label='Centroid')
        ax.legend()

    # title = f"{prompt}, t={timestep}, object={object}\n{loss_type} loss={loss_num} margin={margin} alpha={alpha}\n{block}"
    # ax.set_title(title, loc='center', wrap=True)

    ax.set_title(object)
    ax.axis("off")

    # attn_folder = f"attn_{prompt}_100_pt"
    # os.makedirs(attn_folder, exist_ok=True)

    fig_name = f"t={timestep} object={object} {block} {prompt}"
    # filename = os.path.join(attn_folder, f"{fig_name}.pt")
    # torch.save(edit, filename)

    fig.savefig(os.path.join(save_dir, f"{fig_name}.png"))
    matplotlib.pyplot.close()


def merge_two_images(save_dir, img1_path, img2_path):
    image1 = Image.open(img1_path)
    image2 = Image.open(img2_path)

    h = min(image1.height, image2.height)
    image1 = image1.resize((int(image1.width * h / image1.height), h))
    image2 = image2.resize((int(image2.width * h / image2.height), h))

    new_width = image1.width + image2.width
    new_image = Image.new("RGB", (new_width, h))

    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1.width, 0))

    img_title = img2_path.split("\\")[-1][5:]

    os.makedirs(save_dir, exist_ok=True)
    img_title = os.path.join(save_dir, img_title)

    if not os.path.exists(img_title):
        new_image.save(img_title)
    else:
        print("here")
    # new_image.show()


def merge_three_images(save_dir, img1_path, img2_path, img3_path):
    image1 = Image.open(img1_path)
    image2 = Image.open(img2_path)
    image3 = Image.open(img3_path)

    h = min(image1.height, image2.height, image3.height)
    image1 = image1.resize((int(image1.width * h / image1.height), h))
    image2 = image2.resize((int(image2.width * h / image2.height), h))
    image3 = image3.resize((int(image3.width * h / image3.height), h))

    new_width = image1.width + image2.width + image3.width
    new_image = Image.new("RGB", (new_width, h))

    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1.width, 0))
    new_image.paste(image3, (image1.width + image2.width, 0))

    img_title = img2_path.split("\\")[-1][5:]

    os.makedirs(save_dir, exist_ok=True)
    img_title = os.path.join(save_dir, img_title)

    if not os.path.exists(img_title):
        new_image.save(img_title)
    else:
        print("here")
