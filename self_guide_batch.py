import torch
import torch.nn.functional as F
from self_guidance import plot_attention_map


import numbers
from torch import nn
import math


class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight.to(input.dtype), groups=self.groups)


class Splign:
    @staticmethod
    def _centroid_sg(a):
        # print("attn shape _centroid: ", a.shape) # [1, 64, 64, 1]
        # print("min", torch.min(a).item(), "max",  torch.max(a).item())

        # define the weights - difference is that range is from 0 to 1 instead of from 0 to h-1
        x = torch.linspace(0, 1, a.shape[-2]).to(a.device)  # [64]
        y = torch.linspace(0, 1, a.shape[-3]).to(a.device)
        # a is (n, h, w, k)
        attn_x = a.sum(-3)  # (n, w, k)
        attn_y = a.sum(-2)  # (n, h, k)

        # print("attn_x shape: ", attn_x.shape) # [1, 64, 1]
        # print("attn_y shape: ", attn_y.shape)

        def f(_attn, _linspace):
            # normalize by the sum -> same
            _attn = _attn / (_attn.sum(-2, keepdim=True) + 1e-4)  # [1, 64, 1]
            # print("attn shape", _attn.shape)
            # print("min norm", torch.min(_attn).item(), "max norm", torch.max(_attn).item())
            _weighted_attn = (
                    _linspace[None, ..., None] * _attn
            )  # (n, h or w, k)
            # print("weighted attn: ", _weighted_attn)
            return _weighted_attn.sum(-2)  # (n, k)

        centroid_x = f(attn_x, x)
        # print("centroid_x: ", centroid_x)
        centroid_y = f(attn_y, y)
        # print("centroid_y: ", centroid_y)
        centroid = torch.stack((centroid_x, centroid_y), -1)  # (n, k, 2)
        return centroid

    @staticmethod
    def _centroid(attn_map, smoothing=False, object=None):
        # print(attn_map.shape) [1,64,64,1]
        H, W = attn_map.shape[-3], attn_map.shape[-2]

        # # Applying softmax
        # B, H, W, T = attn_map.shape
        # attn_flat = attn_map.view(B, -1)
        # temperature = 0.1
        # attn_norm = F.softmax(attn_flat / temperature, dim=-1)
        # attn_map = attn_norm.view(B, H, W, T)

        if smoothing:
            kernel_size = 3
            sigma = 0.5
            smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
            input = attn_map.permute(0, 3, 1, 2)
            input = F.pad(input, (1, 1, 1, 1), mode='reflect')
            attn_map = smoothing(input)
            attn_map = attn_map.permute(0, 2, 3, 1)

        x_coords = torch.linspace(0, 1, W).to(attn_map.device)
        y_coords = torch.linspace(0, 1, H).to(attn_map.device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')  # [64,64]

        attn_map = attn_map.squeeze()
        # threshold = attn_map.max() * 0.75  # 75% of the max value
        threshold = attn_map.mean()
        mask = attn_map >= threshold
        masked_attn_map = attn_map * mask

        total = masked_attn_map.sum()
        x_centroid = (masked_attn_map * x_grid).sum() / total
        y_centroid = (masked_attn_map * y_grid).sum() / total
        centroid = torch.stack([x_centroid, y_centroid], dim=-1)  # [2]
        # print(f"centroid {object}", centroid, centroid.shape)

        return centroid

    @staticmethod
    def compute_loss(L2, observed, target):
        if L2:
            loss = (0.5 * (observed - target) ** 2).mean()
        else:
            loss = (observed - target).abs().mean()
        return loss

    @staticmethod
    def centroid(attn, batch_index, i, loss_type=None, loss_num=None, tgt=None, shifts=(0., 0.), objects=[],
                 relative=False, idxs=None, L2=False, module_name=None,
                 cluster_objects=False, prompt=None, relationship="other",
                 alpha=1, margin=0.5, logger=None, self_guidance_mode=False, plot_centroid=False,
                 two_objects=False, centroid_type="sg", img_id=None, smoothing=False, masked_mean=False, object_presence=False,
                 masked_mean_thresh=0.1, masked_mean_weight=0.5, use_energy=False, leaky_relu_slope=0.05, img_num=0):
        # print("attn len: ", len(attn))
        timestep = i
        attn = attn[i]
        # print(f"attn shape at timestep {i}: ", attn.shape) # [batch_size, 64, 64, 1]
        attn = attn[batch_index:batch_index + 1]
        # print(f"attn shape of batch index {batch_index}: ", attn.shape)  # [1, 64, 64, 1]

        # print("shifts", shifts)

        tgt_attn = tgt[i].to(attn.device) if tgt is not None else None
        # print(objects)

        if relative:
            assert tgt_attn is not None
        tgt_attn = tgt_attn if tgt_attn is not None else attn

        # extract the attention map for the specific token
        # if idxs is not None:
        #     attn = attn[..., idxs]  # sd1.5 - [1, 64, 64, 2] -> 2 tokens
        #     tgt_attn = tgt_attn[..., idxs]
        # # print(idxs)

        losses = []
        attn_map_list = []
        if two_objects:
            object_token_mapping = []
            current_idx = 0

            for obj in objects:
                num_words = len(obj.split())
                obj_indices = idxs[current_idx:current_idx + num_words]
                object_token_mapping.append(obj_indices)
                current_idx += num_words

            centroids = []
            energies = []
            for i, obj_indices in enumerate(object_token_mapping):
                combined_attn_map = None
                for idx in obj_indices:
                    attn_map = attn[..., idx:idx + 1]
                    energy = (attn_map ** 2).mean()
                    energies.append(energy)
                    
                    if combined_attn_map is None:
                        combined_attn_map = attn_map
                    else:
                        combined_attn_map = combined_attn_map + attn_map

                if len(obj_indices) > 1:
                    combined_attn_map /= len(obj_indices)

                if centroid_type == "sg":
                    obs_centroid = Splign._centroid_sg(combined_attn_map)
                    obs_centroid = obs_centroid.mean(1, keepdim=True)
                elif centroid_type == "mean":
                    obs_centroid = Splign._centroid(combined_attn_map, smoothing=smoothing)

                attn_map_list.append(combined_attn_map)
                centroids.append(obs_centroid)
                # print(f"centroid {objects[i]}", obs_centroid)

                if plot_centroid:
                    plot_attention_map(combined_attn_map, timestep + 1, module_name,
                                       object=objects[i], # centroid=obs_centroid, 
                                       loss_type=loss_type, loss_num=loss_num,
                                       prompt=prompt, margin=margin, alpha=alpha,
                                       attn_folder="attention_maps", img_id=img_id, img_num=img_num)

                if self_guidance_mode:
                    shift = torch.tensor(shifts[i]).to(attn.device)
                    tgt_centroid = shift.reshape((1,) * (obs_centroid.ndim - shift.ndim) + shift.shape)
                    loss = Splign.compute_loss(L2, obs_centroid, tgt_centroid)
                    losses.append(loss)

            if not self_guidance_mode:
                spatial_loss = Splign.spatial_loss(centroids, relationship, loss_type,
                                                   loss_num, alpha=alpha, margin=margin,
                                                   logger=logger, object_presence=object_presence, 
                                                   leaky_relu_slope=leaky_relu_slope)
                if use_energy:
                    energy_loss = Splign.compute_loss(L2, energies[0], energies[1])
                    spatial_loss = spatial_loss + energy_loss
                
                # MASKED MEAN
                if masked_mean:
                    combined_attn_map = torch.zeros_like(attn_map_list[0])
                    for att_map in attn_map_list:
                        combined_attn_map = combined_attn_map + att_map
                    
                    threshold = combined_attn_map.mean()
                    mask = combined_attn_map >= threshold
                    masked_attn_map = combined_attn_map * mask
            
                    # mean of the masked region only
                    # TODO: sum the two attn maps and use the thresh on that
                    mask_sum = mask.sum().float()
                    if mask_sum > 0:
                        masked_mean = (masked_attn_map.sum() / mask_sum)
            
                        if masked_mean < masked_mean_thresh:
                            # print(objects[i], f"masked mean: {masked_mean.item()}")
                            spatial_loss = spatial_loss + masked_mean_weight * F.relu(masked_mean_thresh - masked_mean)

                # # PREVENT OBJECT OVERLAP
                # loss_contrast = 1 - torch.nn.functional.cosine_similarity(attn_map_list[0].flatten().unsqueeze(dim=0), attn_map_list[1].flatten().unsqueeze(dim=0))
                # spatial_loss = spatial_loss + loss_contrast

                losses.append(spatial_loss)

        else:
            shift = torch.tensor(shifts[0]).to(attn.device)
            obs_centroid = Splign._centroid(attn)

            tgt_centroid = shift.reshape((1,) * (obs_centroid.ndim - shift.ndim) + shift.shape)

            loss = Splign.compute_loss(L2, obs_centroid, tgt_centroid)
            losses.append(loss)

        return losses

    @staticmethod
    def spatial_loss(centroids, relationship, loss_type, loss_num, alpha=1., margin=0.1,
                     lambda_param=2, logger=None, object_presence=False, leaky_relu_slope = 0.05):
        obj1_center = centroids[0].squeeze()
        obj1_x = obj1_center[0]  # .item()
        obj1_y = obj1_center[1]  # .item()
        obj2_center = centroids[1].squeeze()
        obj2_x = obj2_center[0]  # .item()
        obj2_y = obj2_center[1]  # .item()

        if logger:
            logger.info("x1: %s | x2: %s", obj1_x.item(), obj2_x.item())
            logger.info("y1: %s | y2: %s", obj1_y.item(), obj2_y.item())

        loss = 0
        max_margin = 0.9
        if relationship in ["to the left of", "to the right of", "on the left of", "on the right of", "left of",
                            "right of", "near", "on side of", "next to"]:
            if relationship in ["to the left of", "on the left of", "left of", "near", "on side of"]:
                if loss_type == "sigmoid":
                    difference_x = obj1_x - obj2_x
                elif loss_type in ["relu", "squared_relu", "gelu", "leaky_relu"]:
                    difference_x = obj2_x - obj1_x

            if relationship in ["to the right of", "on the right of", "right of", "next to"]:
                if loss_type == "sigmoid":
                    difference_x = obj2_x - obj1_x
                elif loss_type in ["relu", "squared_relu", "gelu", "leaky_relu"]:
                    difference_x = obj1_x - obj2_x
                    # print("difference_x", difference_x)

            if loss_type == "sigmoid":
                loss_horizontal = torch.sigmoid(alpha * difference_x)
            elif loss_type == "relu":
                loss_horizontal = F.relu(alpha * (margin - difference_x))
            elif loss_type == "leaky_relu":
                loss_horizontal = F.leaky_relu(alpha * (margin - difference_x), negative_slope=leaky_relu_slope)
            elif loss_type == "squared_relu":
                loss_horizontal = F.relu(alpha * (margin - difference_x)) ** 2
            elif loss_type == "gelu":
                loss_horizontal = F.gelu(alpha * (margin - difference_x))

            object_presence_loss = F.relu(torch.abs(difference_x) - max_margin)

            if logger:
                logger.info("difference_x: %s | difference_x (scaled): %s | loss_horizontal: %s", difference_x.item(),
                            (alpha * difference_x).item(), loss_horizontal.item())

            difference_y = obj1_y - obj2_y
            loss_vertical_1 = abs(difference_y)
            loss_vertical_2 = 0.5 * difference_y ** 2

            if loss_num == 1:
                loss = loss_horizontal
            if loss_num == 2:
                loss = loss_horizontal + loss_vertical_1
            if loss_num == 3:
                loss = loss_horizontal + loss_vertical_2
            # loss = loss_horizontal + lambda_param * loss_vertical_1
            # loss = loss_horizontal + lambda_param * loss_vertical_2

        if relationship in ["above", "below", "on the top of",
                            "on the bottom of"]:  # "above" in relationship or "top" in relationship or "below" in relationship or "bottom" in relationship:
            if relationship in ["above", "on the top of"]:
                if loss_type == "sigmoid":
                    difference_y = obj1_y - obj2_y  # y increases downwards in the image
                elif loss_type in ["relu", "squared_relu", "gelu", "leaky_relu"]:
                    difference_y = obj2_y - obj1_y

            if relationship in ["below", "on the bottom of"]:
                if loss_type == "sigmoid":
                    difference_y = obj2_y - obj1_y
                elif loss_type in ["relu", "squared_relu", "gelu", "leaky_relu"]:
                    difference_y = obj1_y - obj2_y

            if loss_type == "sigmoid":
                loss_vertical = torch.sigmoid(alpha * difference_y)
            elif loss_type == "relu":
                loss_vertical = F.relu(alpha * (margin - difference_y))
            elif loss_type == "leaky_relu":
                loss_vertical = F.leaky_relu(alpha * (margin - difference_y), negative_slope=leaky_relu_slope)
            elif loss_type == "squared_relu":
                loss_vertical = F.relu(alpha * (margin - difference_y)) ** 2
            elif loss_type == "gelu":
                loss_vertical = F.gelu(alpha * (margin - difference_y))
            
            object_presence_loss = F.relu(torch.abs(difference_y) - max_margin)

            if logger:
                logger.info("difference_y: %s | difference_y (scaled): %s | loss_vertical: %s", difference_y.item(),
                            (alpha * difference_y).item(), loss_vertical.item())

            difference_x = obj1_x - obj2_x
            loss_horizontal_1 = abs(difference_x)
            loss_horizontal_2 = 0.5 * difference_x ** 2

            if loss_num == 1:
                loss = loss_vertical
            if loss_num == 2:
                loss = loss_vertical + loss_horizontal_1
            if loss_num == 3:
                loss = loss_vertical + loss_horizontal_2
            # loss = loss_vertical + lambda_param * loss_horizontal_1 # 4
            # loss = loss_vertical + lambda_param * loss_horizontal_2 # 5
        if object_presence:
            loss = loss + object_presence_loss
        return loss
