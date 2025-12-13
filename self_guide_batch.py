import torch
import torch.nn.functional as F
from utils.vis_utils import plot_attention_map
#import wandb


import numbers
from torch import nn
import math

Energy_A=[]
Energy_B=[]

def entropy(p: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Computes the Shannon entropy of a tensor along a given dimension.

    Args:
        x (torch.Tensor): Input tensor (probabilities or unnormalized values).
        dim (int): Dimension along which to compute entropy.
        eps (float): Small constant to avoid log(0).

    Returns:
        torch.Tensor: Entropy values.
    """
    # Convert to probabilities if not already
    #p = x# / (x.sum(dim=dim, keepdim=True) + eps)
    
    # Compute entropy
    ent = -torch.max(p * torch.log(p + eps))
    return ent

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
    def _centroid(attn_map, probs=1, smoothing=False, energy_loss=None, object=None, energy_coef=0.01,use_max=False):
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

        attn_map = attn_map.squeeze()#*probs.detach()
        # threshold = attn_map.max() * 0.75  # 75% of the max value
        #threshold = attn_map.mean()*0
        #mask = attn_map >= threshold
        masked_attn_map = attn_map#*probs.detach()# * mask
        #print(probs)
        total = masked_attn_map.sum()
        #print(total)
        x_centroid = (masked_attn_map * x_grid).sum() / total
        y_centroid = (masked_attn_map * y_grid).sum() / total
        centroid = torch.stack([x_centroid, y_centroid], dim=-1)  # [2]
        #print("centroid",centroid)

        if use_max:
            mode_normal=torch.unravel_index(torch.argmax(attn_map),attn_map.shape)
            x_centroid=mode_normal[0]/W
            y_centroid=mode_normal[1]/H
            centroid=torch.stack([mode_normal[0]/W, mode_normal[1]/H], dim=-1)


        #print(mode_normal,attn_map.shape)
        #initial 
        #print("centroid",centroid)
        #print("argmax",torch.argmax(attn_map))
        # print(f"centroid {object}", centroid, centroid.shape)
        #volume= torch.sigmoid(attn_map).mean()

        x_s=((x_grid-x_centroid))**2
        y_s=((y_grid-y_centroid))**2

        #x_ss=((x_grid-x_centroid))**2
        #y_ss=((y_grid-y_centroid))**2
        variance=(attn_map*(x_s+y_s)).sum()/attn_map.sum()#+0.00001 #torch.sqrt(((masked_attn_map-masked_attn_map.mean())**2)).mean()
        #energy_coef=0.01
        #print ("variance_old",(attn_map*(x_ss+y_ss)).mean())
        #print ("variance1",variance)
        #print ("variance2",(attn_map*(x_s+y_s)).sum())
        #print("log M",(torch.log(attn_map.max()/attn_map.mean())))
        #print("sigm()", torch.sigmoid((torch.log(attn_map.max()))))
        #print("M",attn_map.max())
        var=2*torch.log(variance)*energy_coef
        volume=(x_s+y_s); energy_coef=1/64
        ent=entropy(masked_attn_map)

        #print(torch.log(attn_map.max())/(2*variance))

        if energy_loss=="log":
            energy=-(torch.log((attn_map).max()))*energy_coef
            #energy=(attn_map.max()*torch.exp(-(x_s+y_s)/2*(variance))).sum()
        elif energy_loss=="entropy":
            energy=entropy(masked_attn_map)
        elif energy_loss=="prob":
            energy=entropy(probs)
        elif energy_loss=="mean":
            energy=attn_map.mean()   
            #energy=(attn_map**2).min()
            #volume= masked_attn_map.min()
        elif energy_loss=="var":
            energy=variance
            #energy=(attn_map**2).mean()
            #volume= masked_attn_map.mean()
        elif energy_loss=="gibs":
            #print(attn_map.min(), attn_map.max())
            energy=-(torch.log(attn_map.max()))*energy_coef+(variance)#*energy_coef
            #volume= masked_attn_map.max()
            #print("max")
        elif energy_loss=="gibsent":
            energy=ent+(variance)
        elif energy_loss=="lin":
            energy=-attn_map.max()#*energy_coef
            #energy=(attn_map**2).sum()
            #volume= masked_attn_map.sum()
        elif energy_loss=="square":
            energy=attn_map.max()**2*energy_coef
        elif energy_loss=="exp":
            energy=torch.exp(attn_map.max())*energy_coef
        else:
            energy=0*var
            #volume=0


        #torch.norm(attn_map, p=1)/total#masked_attn_map.sum()#
        # print(f"centroid {object}", centroid, centroid.shape)


        return centroid,energy, volume,variance, ent

    @staticmethod
    def compute_loss(L2, observed, target):
        if L2:
            loss = (0.5 * (observed - target) ** 2).mean()
        else:
            loss = (observed - target).abs().mean()
        return loss

    @staticmethod
    def centroid(attn, batch_index, i, energy_loss, strategy,loss_type=None, loss_num=None, tgt=None, shifts=(0., 0.), objects=[],
                 relative=False, idxs=None, L2=False, module_name=None,
                 cluster_objects=False, prompt=None, relationship="other",
                 alpha=1, margin=0.5, logger=None, self_guidance_mode=False, plot_centroid=False,
                 two_objects=False, centroid_type="sg", img_id=None, smoothing=False, masked_mean=False, object_presence=False,
                 masked_mean_thresh=0.1, masked_mean_weight=0.5, use_energy=False, leaky_relu_slope=0.05, img_num=0,
                 top_strategy= None, top_loss= None,
                 plotloss=False,lambda_spatial=0.0, lambda_presence=0.0, lambda_balance=0.0):
        # print("attn len: ", len(attn))
        lambda_energy=0
        timestep = i
        attn = attn[i]#; print(attn.shape)
        
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
        #print(gamma)
        losses = []
        attn_map_list = []
        #print(f"strategy: {strategy}, energy_loss: {energy_loss}, gamma:{gamma}")
        #if strategy is not None and strategy!="":
        if "blocks.1" in module_name:
            energy_loss=top_loss#'var'#energy_losses[0]#;'var'
            strategy=top_strategy#'dec'#strategies[0]#'dec'
            lambda_energy=lambda_presence
        else:
            lambda_energy= lambda_balance
        #else:
        #    energy_loss=energy_losses[-1]#;'var'
        #    strategy=strategies[-1]#'dec'
        #else:
        #    strategy=None
        #    energy_loss=None

        #print("strategy:",strategy,"energy_loss:",energy_loss,"lambda_energy:",lambda_energy,"lambda_spatial",lambda_spatial)
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
            volumes=[]
            variances=[]
            entropies=[]
            for i, obj_indices in enumerate(object_token_mapping):
                combined_attn_map = None
                for idx in obj_indices:
                    attn_map = attn[..., idx:idx + 1]
                    #energy = (attn_map ** 2).mean()
                    #energies.append(energy)
                    
                    if combined_attn_map is None:
                        combined_attn_map = attn_map
                    else:
                        combined_attn_map = combined_attn_map + attn_map
                    

                if len(obj_indices) > 1:
                    combined_attn_map /= len(obj_indices)
                attn_map_list.append(combined_attn_map)
                if plot_centroid:
                    plot_attention_map(combined_attn_map, timestep + 1, module_name,
                                       object=objects[i],  #centroid=obs_centroid, 
                                       loss_type=loss_type, loss_num=loss_num,
                                       prompt=prompt, margin=margin, alpha=alpha,
                                       attn_folder="attention_maps", img_id=img_id)#, img_num=img_num)

            aggregate_attention =sum(attn_map_list)
            for combined_attn_map in attn_map_list:
                normalized_attn_map=(combined_attn_map/aggregate_attention )
                #π_1=probs.sum()
                #π_2=1-π_1
                #P_1=torch.log(π_1)-torch.log(2*torch.pi *variance)-var/(variance*2)
                #P_2=torch.log(π_2)-torch.log(2*torch.pi *variance)-var/(variance*2)

                if centroid_type == "sg":
                    obs_centroid = Splign._centroid_sg(combined_attn_map)
                    obs_centroid = obs_centroid.mean(1, keepdim=True)
                elif centroid_type == "mean":
                    obs_centroid,obj_energy,obj_vol,obj_var,obj_ent = Splign._centroid(combined_attn_map,normalized_attn_map, smoothing=smoothing, energy_loss=energy_loss)

                
                centroids.append(obs_centroid)
                energies.append(obj_energy)
                volumes.append(obj_vol)
                variances.append(obj_var)
                entropies.append(obj_ent)
                # print(f"centroid {objects[i]}", obs_centroid)
            """for i, obj_indices in enumerate(object_token_mapping):


                if plot_centroid:
                    plot_attention_map(combined_attn_map, timestep + 1, module_name,
                                       object=objects[i],  #centroid=obs_centroid, 
                                       loss_type=loss_type, loss_num=loss_num,
                                       prompt=prompt, margin=margin, alpha=alpha,
                                       attn_folder="attention_maps", img_id=img_id)#, img_num=img_num)

                if self_guidance_mode:
                    shift = torch.tensor(shifts[i]).to(attn.device)
                    tgt_centroid = shift.reshape((1,) * (obs_centroid.ndim - shift.ndim) + shift.shape)
                    loss = Splign.compute_loss(L2, obs_centroid, tgt_centroid)
                    losses.append(loss)"""
            ###################################E-step
            π_1=attn_map_list[0].sum()
            π_2=attn_map_list[1].sum()
               
            #P_1=append(torch.log(π_1)-torch.log(2*torch.pi *variances[0])torch.exp(-volumes[0]/(variances[0]*2)))
            #P_2=torch.log(π_2)-torch.log(2*torch.pi *variances[1])torch.exp(volumes[1]/(variances[1]*2))
            P_1=(π_1/(2*torch.pi*variances[0]))*torch.exp(-volumes[0]/(variances[0]*2))
            P_2=(π_2/(2*torch.pi *variances[1]))*torch.exp(-volumes[1]/(variances[1]*2))


            R_1=P_1/(P_1+P_2)
            R_2=P_2/(P_1+P_2)

            ###################################M-step

            #print(energies)
            #print("Configuration: energy_loss=", energy_loss, "strategy=", strategy)
            rho=[]
            rho.append(energies[0]/volumes[0])
            rho.append(energies[1]/volumes[1])
            #wandb.log({energy_loss+"_A": energies[0],energy_loss+"_B":energies[1]},step=i)
            #Energy_A.append(energies[0])
            #Energy_B.append(energies[1])


            if plotloss:
                file1 = open("obj1.txt", "a")
                file1.write(str(energies[0].cpu().item())+'\n')
                file1.close()
                file2 = open("obj2.txt", "a")
                file2.write(str(energies[1].cpu().item())+'\n')
                file2.close()
            
            if strategy=="diff":
                #Energy_Loss=Splign.compute_loss(L2,energies[0],energies[1])
                Energy_Loss=(energies[0]- energies[1]).abs().mean()

                #losses.append(Energy_Loss)
                #losses.append(Splign.compute_loss(L2,volumes[0],volumes[1]))
                #losses.append(Splign.compute_loss(L2,rho[0],rho[1]))
                #losses.append(Splign.compute_loss(L2,variances[0],variances[1]))


            elif strategy=="inc":
                #Energy_Loss=-Splign.compute_loss(L2,0, (energies[0]))-Splign.compute_loss(L2,0, (energies[1]))
                Energy_Loss=-energies[0]-energies[1]

                #losses.append(Energy_Loss)
                #losses.append(-Splign.compute_loss(L2,0, (energies[0]))-Splign.compute_loss(L2,0, (energies[1])))
                #losses.append(-Splign.compute_loss(L2,0, (volumes[0]))-Splign.compute_loss(L2,0, (volumes[1])))
                #losses.append(-Splign.compute_loss(L2,0, (rho[0]))-Splign.compute_loss(L2,0, (rho[1])))
                #losses.append(Splign.compute_loss(L2,variances[0],variances[1]))


            elif strategy=="std":
                Energy_Loss=(Splign.compute_loss(L2,energies[0], (energies[0]+energies[1])/2))+Splign.compute_loss(L2,energies[1], (energies[0]+energies[1])/2)
                #losses.append(Energy_Loss)
                #losses.append((Splign.compute_loss(L2,energies[0], (energies[0]+energies[1])/2))+Splign.compute_loss(L2,energies[1], (energies[0]+energies[1])/2))
                #losses.append((Splign.compute_loss(L2,volumes[0], (volumes[0]+volumes[1])/2))+Splign.compute_loss(L2,volumes[1], (volumes[0]+volumes[1])/2))
                #losses.append((Splign.compute_loss(L2,rho[0], (rho[0]+rho[1])/2))+Splign.compute_loss(L2,rho[1], (rho[0]+rho[1])/2))

            elif strategy=="second":
                Energy_Loss=-Splign.compute_loss(L2,energies[1], 0)
                #losses.append(Energy_Loss)
                #losses.append(-Splign.compute_loss(L2,volumes[1], 0))
                #losses.append(-Splign.compute_loss(L2,rho[1], 0))
                #losses.append(-Splign.compute_loss(L2,variances[1], 0))

            elif strategy=="dec":
                #Energy_Loss=Splign.compute_loss(L2,0, (energies[0]))+Splign.compute_loss(L2,0, (energies[1]))
                Energy_Loss=energies[0]+energies[1]

                #losses.append(Energy_Loss)
                #losses.append(Splign.compute_loss(L2,0, (volumes[0]))+Splign.compute_loss(L2,0, (volumes[1])))
                #losses.append(-Splign.compute_loss(L2,0, (rho[0]))-Splign.compute_loss(L2,0, (rho[1])))
                #losses.append(Splign.compute_loss(L2,variances[0],variances[1]))

            elif strategy=="both":
                dec_loss=energies[0]+energies[1]#Splign.compute_loss(L2,0, (energies[0]))+Splign.compute_loss(L2,0, (energies[1]))
                diff_loss=(energies[0]- energies[1]).abs().mean()#Splign.compute_loss(L2,energies[0],energies[1])
                #print(diff_loss/dec_loss)
                Energy_Loss=dec_loss+diff_loss
                #losses.append(Energy_Loss)
                #losses.append(Splign.compute_loss(L2,0, (volumes[0]))+Splign.compute_loss(L2,0, (volumes[1])))
                #losses.append(-Splign.compute_loss(L2,0, (rho[0]))-Splign.compute_loss(L2,0, (rho[1])))
                #losses.append(Splign.compute_loss(L2,variances[0],variances[1]))
            elif strategy=="entgibs":
                dec_loss=Splign.compute_loss(L2,0, (variances[0]))+Splign.compute_loss(L2,0, (variances[1]))
                diff_loss=Splign.compute_loss(L2,energies[0],energies[1])
                #print(diff_loss/dec_loss)
                Energy_Loss=dec_loss+diff_loss
            elif strategy=="mutual":
                Energy_Loss=((1-attn_map_list[1])*entropies[0]+(1-attn_map_list[0])*entropies[1]).mean()#-entropy(attn_map_list[0]*attn_map_list[1])
                
            
            
            else:
                Energy_Loss=0*energies[0]
            if plotloss:
                file3 = open("objs.txt", "a")
                if strategy=="":
                    file3.write(str(0)+'\n')
                else:

                    file3.write(str(Energy_Loss.cpu().item())+'\n')
                file3.close()


            #losses.append(-energies[0])
            #min_energy=torch.min (energies[0],energies[1])
            #max_vol=  torch.max(volumes[0],volumes[1])
            #losses.append(max_vol)#Splign.compute_loss(L2, energies[0], (energies[1]))## change L2 to norm 2 
            #losses.append(Splign.compute_loss(L2, energies[1], (energies[0]+energies[1])/2)*10)
            #losses.append(-Splign.compute_loss(L2,0, (energies[0]+energies[1])/2))
            if not self_guidance_mode:
                if loss_type is None:
                    spatial_loss=0
                else:
                    spatial_loss = Splign.spatial_loss(centroids, relationship, loss_type,
                                                    loss_num, alpha=alpha, margin=margin,
                                                    logger=logger, object_presence=object_presence, 
                                                    leaky_relu_slope=leaky_relu_slope,variances=variances)
                #if use_energy:
                #    ###
                #    energy_loss = Splign.compute_loss(L2, energies[0], energies[1])
                #    spatial_loss = spatial_loss + energy_loss
                
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
                # spatial_loss = spatial_loss + loss_contrastlambda_spatial=lambda_spatial, lambda_presence=lambda_presence, lambda_balance=lambda_balance
                #print("Spatial Loss:",spatial_loss, "Energy_Loss:",Energy_Loss)

                
                total_loss = lambda_spatial*spatial_loss +lambda_energy*Energy_Loss

                losses.append(total_loss)
                #file4 = open("spatial.txt", "a")

                #file4.write(str(spatial_loss.cpu().item())+'\n')
                #ile4.close()


        else:
            shift = torch.tensor(shifts[0]).to(attn.device)
            obs_centroid = Splign._centroid(attn)

            tgt_centroid = shift.reshape((1,) * (obs_centroid.ndim - shift.ndim) + shift.shape)

            loss = Splign.compute_loss(L2, obs_centroid, tgt_centroid)
            losses.append(loss)

        torch.cuda.empty_cache()


        return losses

    @staticmethod
    def spatial_loss(centroids, relationship, loss_type, loss_num, alpha=1., margin=0.1,
                     lambda_param=2, logger=None, object_presence=False, leaky_relu_slope = 0.05,variances=None, lambda_sep=0):
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
            if relationship in ["to the left of", "on the left of", "left of"]:#, "near", "on side of"]:
                difference_x = obj2_x - obj1_x
                #if loss_type == "sigmoid":
                #    difference_x = obj1_x - obj2_x
                #elif loss_type in ["relu", "squared_relu", "gelu", "leaky_relu"]:
                #    difference_x = obj2_x - obj1_x

            if relationship in ["to the right of", "on the right of", "right of"]:#, "next to"]:
                difference_x = obj1_x - obj2_x
                #if loss_type == "sigmoid":
                #    difference_x = obj2_x - obj1_x
                #elif loss_type in ["relu", "squared_relu", "gelu", "leaky_relu"]:
                #    difference_x = obj1_x - obj2_x
                #    # print("difference_x", difference_x)
            if relationship in ["on side of", "next to", "near"]:
                difference_x = abs(obj1_x - obj2_x)
            

            if loss_type == "sigmoid":
                loss_horizontal = torch.sigmoid(alpha * (margin-difference_x))
            elif loss_type == "relu":
                loss_horizontal = F.relu(alpha * (margin - difference_x))
            elif loss_type == "leaky_relu":
                loss_horizontal = F.leaky_relu(alpha * (margin - difference_x), negative_slope=leaky_relu_slope)
            elif loss_type == "squared_relu":
                loss_horizontal = F.relu(alpha * (margin - difference_x)) ** 2
            elif loss_type == "gelu":
                loss_horizontal = F.gelu(alpha * (margin - difference_x))
            elif loss_type == "selu":
                loss_horizontal = F.selu(alpha * (margin - difference_x))
            elif loss_type == "softplus":
                loss_horizontal = F.softplus(alpha * (margin - difference_x))
            elif loss_type == "silu":
                loss_horizontal = F.silu(alpha * (margin - difference_x))
            elif loss_type == "hardtanh":
                loss_horizontal = F.hardtanh((alpha * (margin - difference_x)),min_val=0, max_val=1)
            elif loss_type == "linear":
                loss_horizontal = (alpha * (margin - difference_x))
            elif loss_type == "rrelu":
                loss_horizontal = F.rrelu(alpha * (margin - difference_x))
            elif loss_type == "celu":
                loss_horizontal = F.celu(alpha * (margin - difference_x))
            elif loss_type == "logrelu":
                loss_horizontal = torch.log(F.relu(alpha * (margin - difference_x))+1)
            elif loss_type == "elu":
                loss_horizontal = F.elu(alpha * (margin - difference_x))
            elif loss_type == "softsign":
                loss_horizontal = F.softsign(alpha * (margin - difference_x))
            

            object_presence_loss = F.relu(torch.abs(difference_x) - max_margin)

            if logger:
                logger.info("difference_x: %s | difference_x (scaled): %s | loss_horizontal: %s", difference_x.item(),
                            (alpha * difference_x).item(), loss_horizontal.item())

            difference_y = obj1_y - obj2_y
            loss_vertical_1 = abs(difference_y)
            loss_vertical_2 = 0.5 * difference_y ** 2

            if variances is not None:
                sigm=sum(variances)*0.5
                #mahalonobis=torch.exp(-0.5*((difference_x**2)+(difference_y**2))/(sigm**2))
                mahalonobis=-((difference_x**2)+(difference_y**2))/(sigm**2)
            else:
                mahalonobis=0

            if loss_num == 1:
                loss = loss_horizontal+lambda_sep*mahalonobis
            if loss_num == 2:
                loss = loss_horizontal + loss_vertical_1
            if loss_num == 3:
                loss = loss_horizontal + loss_vertical_2
            # loss = loss_horizontal + lambda_param * loss_vertical_1
            # loss = loss_horizontal + lambda_param * loss_vertical_2

        if relationship in ["above", "below", "on the top of",
                            "on the bottom of"]:  # "above" in relationship or "top" in relationship or "below" in relationship or "bottom" in relationship:
            difference_y = obj1_y - obj2_y; loss_horizontal_1=0;loss_horizontal_2=0;loss_vertical=0
            if relationship in ["above", "on the top of"]:
                difference_y = obj2_y - obj1_y
                #if loss_type == "sigmoid":
                #    difference_y = obj1_y - obj2_y  # y increases downwards in the image
                #elif loss_type in ["relu", "squared_relu", "gelu", "leaky_relu"]:
                #    difference_y = obj2_y - obj1_y

            if relationship in ["below", "on the bottom of"]:
                difference_y = obj1_y - obj2_y
                #if loss_type == "sigmoid":
                #    difference_y = obj2_y - obj1_y
                #elif loss_type in ["relu", "squared_relu", "gelu", "leaky_relu"]:
                #    difference_y = obj1_y - obj2_y

            if loss_type == "sigmoid":
                loss_vertical = torch.sigmoid(alpha * (margin-difference_y))
            elif loss_type == "relu":
                loss_vertical = F.relu(alpha * (margin - difference_y))
            elif loss_type == "leaky_relu":
                loss_vertical = F.leaky_relu(alpha * (margin - difference_y), negative_slope=leaky_relu_slope)
            elif loss_type == "squared_relu":
                loss_vertical = F.relu(alpha * (margin - difference_y)) ** 2
            elif loss_type == "gelu":
                loss_vertical = F.gelu(alpha * (margin - difference_y))
            elif loss_type == "selu":
                loss_vertical = F.selu(alpha * (margin - difference_y))
            elif loss_type == "softplus":
                loss_vertical = F.softplus(alpha * (margin - difference_y))
            elif loss_type == "silu":
                loss_vertical = F.silu(alpha * (margin - difference_y))
            elif loss_type == "hardtanh":
                loss_vertical = F.hardtanh((alpha * (margin - difference_y)),min_val=0, max_val=1)
            elif loss_type == "linear":
                loss_vertical = (alpha * (margin - difference_y))
            elif loss_type == "rrelu":
                loss_vertical = F.rrelu(alpha * (margin - difference_y))
            elif loss_type == "celu":
                loss_vertical = F.celu(alpha * (margin - difference_y))
            elif loss_type == "logrelu":
                loss_vertical = torch.log(F.relu(alpha * (margin - difference_y))+1)
            elif loss_type == "elu":
                loss_vertical = F.elu(alpha * (margin - difference_y))
            elif loss_type == "softsign":
                loss_vertical = F.softsign(alpha * (margin - difference_y))
            
            object_presence_loss = F.relu(torch.abs(difference_y) - max_margin)


            if logger:
                logger.info("difference_y: %s | difference_y (scaled): %s | loss_vertical: %s", difference_y.item(),
                            (alpha * difference_y).item(), loss_vertical.item())

            difference_x = obj1_x - obj2_x
            loss_horizontal_1 = abs(difference_x)
            loss_horizontal_2 = 0.5 * difference_x ** 2

            if variances is not None:
                sigm=sum(variances)*0.5
                mahalonobis=-((difference_x**2)+(difference_y**2))/(sigm**2)
            else:
                mahalonobis=0

            if loss_num == 1:
                loss = loss_vertical+lambda_sep*mahalonobis
            if loss_num == 2:
                loss = loss_vertical + loss_horizontal_1
            if loss_num == 3:
                loss = loss_vertical + loss_horizontal_2
            # loss = loss_vertical + lambda_param * loss_horizontal_1 # 4
            # loss = loss_vertical + lambda_param * loss_horizontal_2 # 5
        if object_presence:
            loss = loss + object_presence_loss
        return loss
