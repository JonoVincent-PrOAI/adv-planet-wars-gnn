import torch
from torch_geometric.data import Data as PyGData, Batch as PyGBatch
from torch_geometric.utils import to_dense_batch, add_self_loops

def owner_one_hot_encoding(owner: torch.Tensor, player_id: int) -> torch.Tensor:
    """Convert owner integer to one-hot encoding. Assume Neutral=0, Controlled=1, Opponent=2 (swaps controlled and opponent if needed)"""
    one_hot = torch.nn.functional.one_hot(
        owner.long(), num_classes=3
    )
    # Swap controlled and opponent if needed
    if player_id == 2:
        one_hot = one_hot[..., [0, 2, 1]]
    return one_hot

def preprocess_graph_data(graph_data: list[PyGData], 
                          player_id: int,
                          use_tick: bool,
                          return_mask: bool = True):
    input = PyGBatch.from_data_list(graph_data)
    planet_owners = input.x[:, 0]
    transporter_owners_per_edge = input.edge_attr[:, 0]
    transporter_owners = input.x[:, 3]
    if return_mask:
        source_mask = to_dense_batch(torch.logical_and(planet_owners == player_id, transporter_owners == 0), input.batch, fill_value=False)[0]
        source_mask = torch.cat((torch.ones(input.batch_size, 1, dtype=torch.bool, device=source_mask.device), source_mask), dim=1)
    if use_tick:
        input.x = torch.cat((owner_one_hot_encoding(planet_owners, player_id),
                        input.x[:, 1:-1],
                        input.tick[input.batch].unsqueeze(-1)),
                        dim=-1)
    else:
        input.x = torch.cat((owner_one_hot_encoding(planet_owners, player_id),
                        input.x[:, 1:-1]), dim=-1)
    input.edge_attr = torch.cat((owner_one_hot_encoding(transporter_owners_per_edge, player_id),
                            input.edge_attr[:, 1:]), dim=-1)
    input.edge_index, input.edge_attr = add_self_loops(input.edge_index, input.edge_attr, fill_value='mean')
    if return_mask:
        return input, source_mask
    else:
        return input

def preprocess_graph_data_unbatched(graph_data: list[PyGData],
                                    player_id: int,
                                    use_tick: bool,
                                    return_mask: bool = True):
    inputs, source_masks = [], []
    for data in graph_data:
        planet_owners = data.x[:, 0]
        transporter_owners_per_edge = data.edge_attr[:, 0]
        transporter_owners = data.x[:, 3]
        if return_mask:
            source_mask = torch.logical_and(planet_owners == player_id, transporter_owners == 0)
            source_mask = torch.cat((torch.ones(1, dtype=torch.bool, device=source_mask.device), source_mask), dim=0)
        if use_tick:
            data.x = torch.cat((owner_one_hot_encoding(planet_owners, player_id),
                                    data.x[:, 1:-1],
                                    data.tick.unsqueeze(-1)),
                                    dim=-1)
        else:
            data.x = torch.cat((owner_one_hot_encoding(planet_owners, player_id),
                                    data.x[:, 1:-1]), dim=-1)
        data.edge_attr = torch.cat((owner_one_hot_encoding(transporter_owners_per_edge, player_id),
                                        data.edge_attr[:, 1:]), dim=-1)
        data.edge_index, data.edge_attr = add_self_loops(data.edge_index, data.edge_attr, fill_value='mean')
        if return_mask:
            inputs.append(data)
            source_masks.append(source_mask)
        else:
            inputs.append(data)
    if return_mask:
        return inputs, source_masks
    else:
        return inputs
    
def collate_source_mask(source_mask_list: list[torch.Tensor], length: int = None) -> torch.Tensor:
    '''Given a list of source masks, pad them to the same length and collate into a single tensor'''
    max_len = max(mask.shape[0] for mask in source_mask_list)
    if length is not None:
        max_len = max(max_len, length)
        result = torch.stack([torch.nn.functional.pad(mask, (0, max_len - mask.shape[0]), value=False) for mask in source_mask_list], dim=0)[:, :length]
    else:
        result = torch.stack([torch.nn.functional.pad(mask, (0, max_len - mask.shape[0]), value=False) for mask in source_mask_list], dim=0)   

    return result

