import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from skimage.filters import threshold_otsu
from scipy import ndimage
# from self_guidance import plot_attention_map


class SelfGuidanceEdits:
    @staticmethod
    def _centroid_sg(a):
        # print("attn shape _centroid: ", a.shape) # [1, 64, 64, 1]
        # print("min", torch.min(a).item(), "max",  torch.max(a).item())
        
        # define the weights - difference is that range is from 0 to 1 instead of from 0 to h-1
        x = torch.linspace(0, 1, a.shape[-2]).to(a.device) # [64]
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
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij') # [64,64]

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
    def _attn_diff_norm(report_attn, hard=False, thresh=0.5):
        # print("what is this report_attn.min(2, keepdim=True)[0]", report_attn.min(2, keepdim=True)[0].shape)
        attn_min = report_attn.min(2, keepdim=True)[0].min(3, keepdim=True)[0]
        attn_max = report_attn.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
        attn_thresh = (report_attn - attn_min) / (attn_max - attn_min + 1e-4)
        if hard:
            return (attn_thresh > thresh) * 1.0
        attn_binarized = torch.sigmoid((attn_thresh - thresh) * 10)
        attn_min = attn_binarized.min(2, keepdim=True)[0].min(3, keepdim=True)[0]
        attn_max = attn_binarized.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
        attn_norm = (attn_binarized - attn_min) / (attn_max - attn_min + 1e-4)
        return attn_norm

    @staticmethod
    def compute_loss(L2, observed, target):
        if L2:
            loss = (0.5 * (observed - target) ** 2).mean()
        else:
            loss = (observed - target).abs().mean()
        return loss
    
    @staticmethod
    def appearance(aux, i, tgt, idxs=None, L2=False, thresh=0.5, two_objects=False, self_guidance_mode=False):
        tgt_aux = tgt
        dev = torch.utils._pytree.tree_flatten(aux)[0][-1].device            
        new_aux = {}
        for k, v in aux.items():
            first_value = next(iter(v.values()))
            new_aux[k] = first_value[i]
        aux = new_aux

        tgt_aux = {k: next(iter(v.values()))[i].detach().to(dev) for k, v in tgt_aux.items()}
        
        if idxs is not None:
            aux['last_attn'] = aux['last_attn'][..., idxs]
            tgt_aux['last_attn'] = tgt_aux['last_attn'][..., idxs]

        def _compute(last_feats, last_attn):
            last_attn = last_attn.detach()
            last_feats = last_feats.permute(0, 2, 3, 1)

            last_attn = SelfGuidanceEdits._attn_diff_norm(last_attn, hard=True, thresh=thresh)
            last_attn = TF.resize(last_attn.permute(0, 3, 1, 2), last_feats.shape[1], antialias=True).permute(0, 2, 3,
                                                                                                              1)  # TODO VERIFY SHAPE 1

            # import pdb; pdb.set_trace()
            app = (last_attn[..., None] * last_feats[..., None, :]).sum((-3, -4)) / (
                        1e-4 + last_attn.sum((-2, -3))[..., None])  # TODO MULTI AXIS SUM LIKE THIS?

            return app

        def _compute_appearance(aux, tgt_aux, L2):
            app = _compute(**aux)
            tgt_app = _compute(**tgt_aux)

            loss = SelfGuidanceEdits.compute_loss(L2, app, tgt_app)
            return loss
        
        # TODO: PROBABLY NO NEED FOR THIS LOOP AND CAN RETURN ONE LOSS VALUE
        losses = []        
        if two_objects:
            for i in range(len(idxs)):
                # print(aux) # dict
                # print(aux['last_attn'].shape) # torch.Size([1, 64, 64, 10])
                aux_obj = {key: value[..., i:i+1] for key, value in aux.items()}
                tgt_aux_obj = {key: value[..., i:i+1] for key, value in tgt_aux.items()}

                loss = _compute_appearance(aux_obj, tgt_aux_obj, L2)
                losses.append(loss)
        else:
            loss = _compute_appearance(aux, tgt_aux, L2)
            losses.append(loss)
            
        # loss = _compute_appearance(aux, tgt_aux, L2)
        # print("loss: ", loss, "losses: ", losses)
        # losses.append(loss)
        return losses

    @staticmethod
    def shape(attn, i, tgt, idxs=None, thresh=True, L2=False, two_objects=False, self_guidance_mode=False):
        attn = attn[i]
        tgt_attn = tgt[i].to(attn.device)

        if idxs is not None:
            attn = attn[..., idxs] # [1, 64, 64, 77] -> [1, 64, 64, 8]
            tgt_attn = tgt_attn[..., idxs]
            
        losses = []
        if two_objects:
            for i in range(len(idxs)):
                attn_map = attn[..., i:i+1]
                tgt_attn_map = tgt_attn[..., i:i+1]
        
                if thresh:
                    attn_map = SelfGuidanceEdits._attn_diff_norm(attn_map)                    
                    tgt_attn_map = SelfGuidanceEdits._attn_diff_norm(tgt_attn_map, hard=True)
        
                loss = SelfGuidanceEdits.compute_loss(L2, attn_map, tgt_attn_map)
            losses.append(loss)
                
        else:
            if thresh:
                attn = SelfGuidanceEdits._attn_diff_norm(attn)
                tgt_attn = SelfGuidanceEdits._attn_diff_norm(tgt_attn, hard=True)
        
            loss = SelfGuidanceEdits.compute_loss(L2, attn, tgt_attn)
            losses.append(loss)
        
        # if thresh:
        #     attn = SelfGuidanceEdits._attn_diff_norm(attn)
        #     tgt_attn = SelfGuidanceEdits._attn_diff_norm(tgt_attn, hard=True)
    
        # loss = SelfGuidanceEdits.compute_loss(L2, attn, tgt_attn)
        # losses.append(loss)
        return losses
            
    @staticmethod
    def centroid(attn, i, loss_type=None, loss_num=None, tgt=None, shifts=(0., 0.), objects=[], 
                 relative=False, idxs=None, L2=False, module_name=None, 
                 cluster_objects=False, prompt=None, relationship="other", 
                 alpha=1, margin=0.5, logger=None, self_guidance_mode=False, plot_centroid=False,
                 two_objects=False):
        timestep = i
        attn = attn[i]
        tgt_attn = tgt[i].to(attn.device) if tgt is not None else None

        if relative: assert tgt_attn is not None
        tgt_attn = tgt_attn if tgt_attn is not None else attn

        # extract the attention map for the specific token
        if idxs is not None:
            attn = attn[..., idxs] # sd1.5 - [1, 64, 64, 2] -> 2 tokens
            tgt_attn = tgt_attn[..., idxs]

        losses = []
        # WE INTRODUCED LOSS += -> RUN AGAIN TO SEE IF THERE IS DIFFERENCE
        # loss = 0
        if two_objects:        
            centroids = []   
            for i in range(len(shifts)):
                shift = torch.tensor(shifts[i]).to(attn.device)                
                attn_map = attn[..., i:i+1]
                
                if cluster_objects:
                    attn_map = SelfGuidanceEdits.cluster_attention_maps(attn_map, method="otsu")
                    
                obs_centroid = SelfGuidanceEdits._centroid(attn_map) # , objects[i] # [1,4,2] when multiple indices -> 4
                # obs_centroid = obs_centroid.mean(1, keepdim=True) # [1,1,2]
                centroids.append(obs_centroid)
                
                # if plot_centroid:
                #     plot_attention_map(attn_map, timestep+1, module_name,
                #                     centroid=obs_centroid, object=objects[i],
                #                     loss_type=loss_type, loss_num=loss_num,
                #                     prompt=prompt, margin=margin, alpha=alpha)

                if self_guidance_mode:
                    tgt_centroid = shift.reshape((1,) * (obs_centroid.ndim - shift.ndim) + shift.shape)

                    loss = SelfGuidanceEdits.compute_loss(L2, obs_centroid, tgt_centroid)
                    losses.append(loss)
                
            if not self_guidance_mode:
                spatial_loss = SelfGuidanceEdits.spatial_loss(centroids, relationship, loss_type, 
                                                              loss_num, alpha=alpha, margin=margin, 
                                                              logger=logger)
                losses.append(spatial_loss)
       
        else:
            shift = torch.tensor(shifts[0]).to(attn.device)
            obs_centroid = SelfGuidanceEdits._centroid(attn)
            
            tgt_centroid = shift.reshape((1,) * (obs_centroid.ndim - shift.ndim) + shift.shape)
            
            loss = SelfGuidanceEdits.compute_loss(L2, obs_centroid, tgt_centroid)
            losses.append(loss)
        
        # losses.append(loss)
        # assert len(losses) == 1, f"len(losses): {len(losses)}"
        return losses

    @staticmethod
    def size(attn, i, tgt=None, relative=False, shifts=(0.,), thresh=True, idxs=None, L2=False, two_objects=False, self_guidance_mode=False):
        attn = attn[i]
        tgt_attn = tgt[i].to(attn.device) if tgt is not None else None

        tgt_attn = tgt_attn if tgt_attn is not None else attn
        
        if idxs is not None:
            attn = attn[..., idxs]
            tgt_attn = tgt_attn[..., idxs]
            
        if thresh:
            def _size(report_attn):
                attn_norm = SelfGuidanceEdits._attn_diff_norm(report_attn)
                return attn_norm.mean((-2, -3))[..., None]
        else:
            def _size(attn):
                return attn.mean((-2, -3))[..., None]
        
        losses = []
        if two_objects:
            spatial_sizes = []
            for i in range(len(shifts)):
                shift = torch.tensor(shifts[i]).to(attn.device)
                attn_map = attn[..., i:i+1]
                size_obs = _size(attn_map)
                spatial_sizes.append(size_obs)                
        
                if self_guidance_mode:
                    size_tgt = shift.reshape((1,) * (size_obs.ndim - shift.ndim) + shift.shape)
                    loss = SelfGuidanceEdits.compute_loss(L2, size_obs, size_tgt)
                    losses.append(loss)
                    
            # if not self_guidance_mode:
            #     spatial_loss = SelfGuidanceEdits.spatial_loss(centroids, relationship, loss_type, loss_num, alpha=alpha, margin=margin, logger=logger)
            #     losses.append(spatial_loss)
        else:
            shift = torch.tensor(shifts[0]).to(attn.device)
            size_obs = _size(attn)
            
            size_tgt = shift.reshape((1,) * (size_obs.ndim - shift.ndim) + shift.shape)
            
            loss = SelfGuidanceEdits.compute_loss(L2, size_obs, size_tgt)
            losses.append(loss)
            
        # print(losses)
        return losses
    
    @staticmethod
    def cluster_attention_maps(attn_maps, method="otsu", percentile=90, enhance_factor=1.0, dampen_factor=0.5):
        # combined_attn = attn_maps.get(target_obj)
        combined_attn = attn_maps

        if method == "otsu":
            # Otsu's method - finds threshold that minimizes intra-class variance
            threshold = threshold_otsu(combined_attn.cpu().detach().numpy())

            # Create initial mask with Otsu threshold
            mask = combined_attn > threshold

            # Identify higher-valued pixels (core regions)
            high_value_threshold = threshold * 1.5  # Adjust this multiplier as needed
            core_regions = combined_attn > high_value_threshold

            # Convert tensors to numpy for spatial processing
            mask_np = mask.cpu().numpy()
            core_np = core_regions.cpu().numpy()

            # Keep only pixels that are spatially close to high-value regions
            # Create distance map from core regions
            distance_map = ndimage.distance_transform_edt(~core_np)
            # Keep only pixels within specified distance of core regions
            proximity_threshold = 5  # number of pixels
            proximity_mask = distance_map <= proximity_threshold

            # Update mask to include only pixels that satisfy both conditions
            refined_mask = mask_np & proximity_mask
            mask = torch.tensor(refined_mask, device=combined_attn.device, dtype=torch.bool)
        elif method == "percentile":
            threshold = torch.quantile(combined_attn.float(), percentile / 100.0).item()
            # from float16 - half-precision to float32 - single-precision -> not changing the values, only the number of decimal places
        elif method == "mean":
            scaling_factor = 1.5
            threshold = combined_attn.mean().item() * scaling_factor
        # elif method == "adaptive":
        #     obj_stats = {obj: {"mean": map.mean().item(), "max": map.max().item()}
        #                 for obj, map in attn_maps.items()}
        #     threshold = obj_stats[target_obj]["mean"] + 0.5 * (obj_stats[target_obj]["max"] - obj_stats[target_obj]["mean"])
        else:
            raise ValueError(f"Unknown method: {method}")

        # print(f"Dynamically determined threshold: {threshold:.4f}")

        # Apply the threshold with enhancement and dampening
        if method != "otsu":
            mask = combined_attn > threshold
        combined_attn = torch.where(mask, combined_attn * enhance_factor, combined_attn * dampen_factor)

        return combined_attn
        
    @staticmethod
    def spatial_loss(centroids, relationship, loss_type, loss_num, alpha=1., margin=0.1, lambda_param=2, logger=None):
        obj1_center = centroids[0].squeeze()
        obj1_x = obj1_center[0] #.item()
        obj1_y = obj1_center[1] #.item()
        obj2_center = centroids[1].squeeze()
        obj2_x = obj2_center[0] #.item()
        obj2_y = obj2_center[1] #.item()
        
        if logger:
            logger.info("x1: %s | x2: %s", obj1_x.item(), obj2_x.item())
            logger.info("y1: %s | y2: %s", obj1_y.item(), obj2_y.item())
        
        loss = 0
        if relationship in ["to the left of", "to the right of", "on the left of", "on the right of", "left of", "right of", "near", "on side of", "next to"]:
            if relationship in ["to the left of", "on the left of", "left of", "near", "on side of"]:
                if loss_type == "sigmoid":
                    difference_x = obj1_x - obj2_x
                elif loss_type in ["relu", "squared_relu", "gelu"]:
                    difference_x = obj2_x - obj1_x
                
            if relationship in ["to the right of", "on the right of", "right of", "next to"]:
                if loss_type == "sigmoid":
                    difference_x = obj2_x - obj1_x
                elif loss_type in ["relu", "squared_relu", "gelu"]:
                    difference_x = obj1_x - obj2_x
                
            if loss_type == "sigmoid":
                loss_horizontal = torch.sigmoid(alpha * difference_x)
            elif loss_type == "relu":
                loss_horizontal = F.relu(margin - alpha * difference_x)
            elif loss_type == "squared_relu":
                loss_horizontal = F.relu(margin - alpha * difference_x) ** 2
            elif loss_type == "gelu":
                loss_horizontal = F.gelu(margin - alpha * difference_x)
            
            if logger: 
                logger.info("difference_x: %s | difference_x (scaled): %s | loss_horizontal: %s", difference_x.item(), (alpha * difference_x).item(), loss_horizontal.item())
            
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
        
        if relationship in ["above", "below", "on the top of", "on the bottom of"]: #"above" in relationship or "top" in relationship or "below" in relationship or "bottom" in relationship:
            if relationship in ["above", "on the top of"]:
                if loss_type == "sigmoid":
                    difference_y = obj1_y - obj2_y # y increases downwards in the image
                elif loss_type in ["relu", "squared_relu", "gelu"]:
                    difference_y = obj2_y - obj1_y
                
            if relationship in ["below", "on the bottom of"]:
                if loss_type == "sigmoid":
                    difference_y = obj2_y - obj1_y
                elif loss_type in ["relu", "squared_relu", "gelu"]:
                    difference_y = obj1_y - obj2_y
                
            if loss_type == "sigmoid":
                loss_vertical = torch.sigmoid(alpha * difference_y)
            elif loss_type == "relu":
                loss_vertical = F.relu(margin - alpha * difference_y)
            elif loss_type == "squared_relu":
                loss_vertical = F.relu(margin - alpha * difference_y) ** 2
            elif loss_type == "gelu":
                loss_vertical = F.gelu(margin - alpha * difference_y)
            
            if logger: 
                logger.info("difference_y: %s | difference_y (scaled): %s | loss_vertical: %s", difference_y.item(), (alpha * difference_y).item(), loss_vertical.item())
            
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
