

        # unflip_proj = target_projection / torch.tan(0.5 * fovy[tgt_idx,None,None]) # Consider focal length / FoV
        
        # import pdb; pdb.set_trace()
        # # projected_loc_norm = unflip_proj.reshape(-1,2).fliplr().reshape(num_multiview-1,-1,2) * torch.tensor([1,-1]).to(device)[None,None,...]
        
        # projected_loc = unflip_proj * torch.tensor([-1,1]).to(device)[None,None,...]
        
        # for i, proj in enumerate(projected_loc):
        #     re_dict["proj_maps"][f"{k}_{i}"] = proj
        
        # # Generate depth mask
        
        # with torch.no_grad():
        #     coord_loc = projected_loc.reshape(-1,2).fliplr().reshape(num_multiview-1,-1,2)
        #     proj_loc = (coord_loc * (img_size / 2) + (img_size / 2)).clamp(0,img_size-0.0001)
        #     proj_pix = proj_loc.int()