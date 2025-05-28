import torch
import torch.nn.functional as F
from self_guidance import plot_attention_map


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
    def _centroid(attn_map, object=None):
        # print(attn_map.shape) [1,64,64,1]
        H, W = attn_map.shape[-3], attn_map.shape[-2]

        x_coords = torch.linspace(0, 1, W).to(attn_map.device)
        y_coords = torch.linspace(0, 1, H).to(attn_map.device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')  # [64,64]

        attn_map = attn_map.squeeze()
        # threshold = attn_map.max() * 0.75  # 75% of the max value
        threshold = attn_map.mean()
        mask = attn_map >= threshold
        masked_attn_map = attn_map * mask

        total = masked_attn_map.sum()
        # print("Attention total: ", total.item())
        if not total.sum() > 0:
            print("Warning: attention map all zero")
            return torch.tensor([0.5, 0.5], device=attn_map.device)

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
                 two_objects=False, centroid_type="sg"):  
        # print("attn len: ", len(attn))
        timestep = i
        attn = attn[i]
        # print(f"attn shape at timestep {i}: ", attn.shape) # [batch_size, 64, 64, 1]
        attn = attn[batch_index:batch_index+1]
        # print(f"attn shape of batch index {batch_index}: ", attn.shape)  # [1, 64, 64, 1]
        
        # print("shifts", shifts)
               
        tgt_attn = tgt[i].to(attn.device) if tgt is not None else None

        if relative: 
            assert tgt_attn is not None
        tgt_attn = tgt_attn if tgt_attn is not None else attn

        # extract the attention map for the specific token
        # if idxs is not None:
        #     attn = attn[..., idxs]  # sd1.5 - [1, 64, 64, 2] -> 2 tokens
        #     tgt_attn = tgt_attn[..., idxs]

        losses = []
        if two_objects:
            object_token_mapping = []
            current_idx = 0

            for obj in objects:
                num_words = len(obj.split())
                obj_indices = idxs[current_idx:current_idx+num_words]
                object_token_mapping.append(obj_indices)
                current_idx += num_words
                
            centroids = []

            for i, obj_indices in enumerate(object_token_mapping):
                # If object has multiple words, combine them into one attention map.
                combined_attn_map = None
                for idx in obj_indices:
                    attn_map = attn[..., idx:idx+1]
                    if combined_attn_map is None:
                        combined_attn_map = attn_map
                    else:
                        combined_attn_map = combined_attn_map + attn_map

                if len(obj_indices) > 1:
                    combined_attn_map /= len(obj_indices)

                # Calculate centroid from attention map
                if centroid_type == "sg":
                    obs_centroid = Splign._centroid_sg(combined_attn_map)
                    obs_centroid = obs_centroid.mean(1, keepdim=True)
                elif centroid_type == "mean":
                    obs_centroid = Splign._centroid(combined_attn_map)
                centroids.append(obs_centroid)
                # print("centroid", obs_centroid, obs_centroid.shape)
                
                if plot_centroid:                    
                    plot_attention_map(combined_attn_map, timestep+1, module_name,
                                    centroid=obs_centroid, object=objects[i],
                                    loss_type=loss_type, loss_num=loss_num,
                                    prompt=prompt, margin=margin, alpha=alpha)

                if self_guidance_mode:
                    shift = torch.tensor(shifts[i]).to(attn.device)
                    tgt_centroid = shift.reshape((1,) * (obs_centroid.ndim - shift.ndim) + shift.shape)

                    loss = Splign.compute_loss(L2, obs_centroid, tgt_centroid)
                    losses.append(loss)

            if not self_guidance_mode:
                spatial_loss = Splign.spatial_loss(centroids, relationship, loss_type,
                                                              loss_num, alpha=alpha, margin=margin,
                                                              logger=logger)
                # print("spatial_loss", spatial_loss)
                losses.append(spatial_loss)

        else:
            shift = torch.tensor(shifts[0]).to(attn.device)
            obs_centroid = Splign._centroid(attn)

            tgt_centroid = shift.reshape((1,) * (obs_centroid.ndim - shift.ndim) + shift.shape)

            loss = Splign.compute_loss(L2, obs_centroid, tgt_centroid)
            losses.append(loss)

        # losses.append(loss)
        # assert len(losses) == 1, f"len(losses): {len(losses)}"
        return losses

    @staticmethod
    def spatial_loss(centroids, relationship, loss_type, loss_num, alpha=1., margin=0.1, lambda_param=2, logger=None):
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
        if relationship in ["to the left of", "to the right of", "on the left of", "on the right of", "left of",
                            "right of", "near", "on side of", "next to"]:
            if relationship in ["to the left of", "on the left of", "left of", "near", "on side of"]:
                # obj_1 < obj_2 is correct, positive diff indicates positive loss
                difference_x = obj1_x - obj2_x 
            elif relationship in ["to the right of", "on the right of", "right of", "next to"]:
                # obj_1 > obj_2 is correct
                difference_x = obj2_x - obj1_x

            if loss_type == "sigmoid":
                loss_horizontal = torch.sigmoid(margin + alpha * difference_x)
            elif loss_type == "relu":
                loss_horizontal = F.relu(margin + alpha * difference_x)
            elif loss_type == "squared_relu":
                loss_horizontal = F.relu(margin + alpha * difference_x) ** 2
            elif loss_type == "gelu":
                loss_horizontal = F.gelu(margin + alpha * difference_x)
            elif loss_type == "linear":
                loss_horizontal = margin + alpha*0.5*difference_x + 0.5
            elif loss_type == "quadratic":
                loss_horizontal = 0.8*difference_x**2 + 0.2*difference_x - 0.1125
            elif loss_type == "double_relu":
                loss_horizontal = F.relu(-(difference_x+0.6)) + 0.7*F.relu(difference_x+0.4)

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
                # obj_1 < obj_2 is correct
                difference_y = obj1_y - obj2_y
            elif relationship in ["below", "on the bottom of"]:
                # obj_1 > obj_2 is correct
                difference_y = obj2_y - obj1_y

            if loss_type == "sigmoid":
                loss_vertical = torch.sigmoid(margin + alpha * difference_y)
            elif loss_type == "relu":
                loss_vertical = F.relu(margin + alpha * difference_y)
            elif loss_type == "squared_relu":
                loss_vertical = F.relu(margin + alpha * difference_y) ** 2
            elif loss_type == "gelu":
                loss_vertical = F.gelu(margin + alpha * difference_y)
            elif loss_type == "linear":
                loss_vertical = margin + alpha*0.5*difference_y + 0.5
            elif loss_type == "quadratic":
                loss_vertical = alpha*difference_y**2 + margin*difference_y - 0.1125
            elif loss_type == "double_relu":
                loss_vertical = F.relu(-(difference_y+0.6)) + 0.7*F.relu(difference_y+0.4)

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
        return loss
