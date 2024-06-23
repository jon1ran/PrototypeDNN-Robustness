
import torch
from adversarial_attacks import PGDL2_attack
from functools import partial
import pandas as pd
import numpy as np


def stability_cost(model, batch_x, perturbed_batch_x, k):
    #K is the number of prototypes considered as the explanation
    
    #Get the latent representation of the prototypes
    prototypes_encoded = model.prototype_layer.prototype_distances
    
    #Get the distances between the examples and the prototypes
    distances = model.prototype_layer(model.encoder(batch_x).view(batch_x.size(0), -1))  
    distances_adv = model.prototype_layer(model.encoder(perturbed_batch_x).view(perturbed_batch_x.size(0), -1))
    
    #Order for each example the prototypes by distance
    #ori_explanation = prototypes_encoded[torch.argsort(distances, dim=1)][:, :k].view(batch_x.size(0), -1)
    #perturbed_explanation = prototypes_encoded[torch.argsort(distances_adv, dim=1)][:, :k].view(batch_x.size(0), -1)
    
    #Get the latent representation of the examples
    ori_h = model.encoder(batch_x).view(batch_x.size(0), -1)
    perturbed_h = model.encoder(perturbed_batch_x).view(batch_x.size(0), -1)
    
    #dif_exp = ori_explanation - perturbed_explanation
    dif_exp = distances - distances_adv
    dif_h = ori_h - perturbed_h
    
    exp_norm = torch.norm(dif_exp, dim=1)
    h_norm = torch.norm(dif_h, dim=1)
    
    r = exp_norm / (h_norm + 1e-10)
    
    return torch.mean(r)
    
    
def model_stability(model, data_loader, eps=4, alpha=0.01, iters=40, k=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define the device
    score = 0
    
    #In case data_loader is a list of data_loaders
    if isinstance(data_loader, list):
        for loader in data_loader:
            for i, batch in enumerate(loader):
                batch_x = batch[0].to(device)
                
                #In this case perturbed_batch_x is batch_x, because of the implementation of the PGDL2_attack.
                #Check the function to notice that it calls the loss_f with batch_x as perturbed_batch_x.
                loss_f = partial(stability_cost, model=model, perturbed_batch_x=batch_x, k=k)
                worst_case_neighbours = PGDL2_attack(batch_x, loss_f, iters, eps, alpha, random_start=True)
                score += loss_f(batch_x=worst_case_neighbours)
    else:
        for i, batch in enumerate(data_loader):
            batch_x = batch[0].to(device)
            
            #In this case perturbed_batch_x is batch_x, because of the implementation of the PGDL2_attack.
            #Check the function to notice that it calls the loss_f with batch_x as perturbed_batch_x.
            loss_f = partial(stability_cost, model=model, perturbed_batch_x=batch_x, k=k)
            worst_case_neighbours = PGDL2_attack(batch_x, loss_f, iters, eps, alpha, random_start=True)
            score += loss_f(batch_x=worst_case_neighbours)
    
    if isinstance(data_loader, list):
        total_length = sum(len(loader) for loader in data_loader)
    else:
        total_length = len(data_loader)
    
    return score / total_length
    

def model_purity(model, data_loader, k):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define the device
    score = 0
    top = None
    #In case data_loader is a list of data_loaders
    if isinstance(data_loader, list):
        for loader in data_loader:
            for i, batch in enumerate(loader):
                batch_x = batch[0].to(device)
                batch_y = batch[1].to(device)
                
                distances = model.prototype_layer(model.encoder(batch_x).view(batch_x.size(0), -1))
                
                # Get the indices of the rows with the smallest distances for each column
                min_indices = torch.argsort(distances, dim=0)[:k, :]
        
                # Get the corresponding distances for each column in min_indices
                min_distances = torch.zeros_like(min_indices, dtype=torch.float32)
                min_classes = torch.zeros_like(min_indices, dtype=torch.float32)
                for j in range(min_indices.size(1)):
                    col = min_indices[:, j]
                    
                    min_distances[:, j] = distances[col, j]
                    min_classes[:, j] = batch_y[col]
                    
                #Combine the two matrix into one, where each element has a tuple of distance and class
                combined = torch.stack((min_distances, min_classes), dim=-1)
                
                if top==None:
                    top = combined
                else:
                    merged_tensor = torch.cat((top, combined), dim=0)

                    # Sort each column based on the first element of each element of each row
                    sorted_tensor, indices = torch.sort(merged_tensor[..., 0], dim=0)

                    # Rearrange the elements in the tensors according to the sorted indices
                    top = torch.gather(merged_tensor, 0, indices.unsqueeze(-1).expand_as(merged_tensor))[:k]
                
    else:
        for i, batch in enumerate(data_loader):
            batch_x = batch[0].to(device)
            batch_y = batch[1].to(device)
            
            distances = model.prototype_layer(model.encoder(batch_x).view(batch_x.size(0), -1))
            
            # Get the indices of the rows with the smallest distances for each column
            min_indices = torch.argsort(distances, dim=0)[:k, :]
    
            # Get the corresponding distances for each column in min_indices
            min_distances = torch.zeros_like(min_indices, dtype=torch.float32)
            min_classes = torch.zeros_like(min_indices, dtype=torch.float32)
            for j in range(min_indices.size(1)):
                col = min_indices[:, j]
                
                min_distances[:, j] = distances[col, j]
                min_classes[:, j] = batch_y[col]
                
            #Combine the two matrix into one, where each element has a tuple of distance and class
            combined = torch.stack((min_distances, min_classes), dim=-1)
            
            if top==None:
                top = combined
            else:
                merged_tensor = torch.cat((top, combined), dim=0)

                # Sort each column based on the first element of each element of each row
                sorted_tensor, indices = torch.sort(merged_tensor[..., 0], dim=0)

                # Rearrange the elements in the tensors according to the sorted indices
                top = torch.gather(merged_tensor, 0, indices.unsqueeze(-1).expand_as(merged_tensor))[:k]
                
    # Get the first element of each element in top
    #We are interested in the class of the top-k closest images for each prototype (column)
    top = top[:, :, 1].cpu().detach().numpy()
    print(top)
    
    df = pd.DataFrame(top)

    # Calculate the percentage of occurrences of the majority class in each column
    purity = df.apply(lambda x: x.value_counts().max() / len(x))
    print(purity)
    return purity.mean().item()
    
    
def mse(imageA, imageB):
    # Compute the mean squared error between two images
    err = np.sum((imageA - imageB) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


        
        
        
    
    