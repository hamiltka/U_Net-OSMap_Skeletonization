import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from advanced_UNet_Model import UNetMultiTask
from create_Dataloaders import get_dataloaders

def get_neighbors(x, y, shape):
    """Return list of 8-connected neighbor coordinates for (x, y) within image bounds."""
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < shape[0] and 0 <= ny < shape[1]:
                neighbors.append((nx, ny))
    return neighbors

def find_nodes_and_valence(skel_mask):
    """
    Returns list of (x, y, valence) for each skeleton pixel in a binary mask.
    """
    nodes = []
    skel = skel_mask.astype(np.uint8)
    for x in range(skel.shape[0]):
        for y in range(skel.shape[1]):
            if skel[x, y]:
                valence = 0
                for nx, ny in get_neighbors(x, y, skel.shape):
                    if skel[nx, ny]:
                        valence += 1
                nodes.append((x, y, valence))
    return nodes

def group_nodes_by_valence(nodes):
    """
    Group nodes by their valence: returns dict {valence: [(x, y), ...]}
    """
    groups = {}
    for x, y, val in nodes:
        if val not in groups:
            groups[val] = []
        groups[val].append((x, y))
    return groups

def bipartite_match(pred_nodes, gt_nodes, max_dist=3):
    """
    Perform bipartite matching between predicted and ground-truth nodes
    using Hungarian algorithm, within max_dist threshold.
    """
    if len(pred_nodes) == 0 or len(gt_nodes) == 0:
        return [], list(range(len(pred_nodes))), list(range(len(gt_nodes)))
    cost_matrix = np.zeros((len(pred_nodes), len(gt_nodes)))
    for i, p in enumerate(pred_nodes):
        for j, g in enumerate(gt_nodes):
            cost_matrix[i, j] = np.linalg.norm(np.array(p) - np.array(g))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    unmatched_pred = set(range(len(pred_nodes)))
    unmatched_gt = set(range(len(gt_nodes)))
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] <= max_dist:
            matches.append((i, j))
            unmatched_pred.discard(i)
            unmatched_gt.discard(j)
    return matches, list(unmatched_pred), list(unmatched_gt)

def evaluate_node_metrics(pred_mask, gt_mask):
    pred_nodes_by_valence = group_nodes_by_valence(find_nodes_and_valence(pred_mask))
    gt_nodes_by_valence = group_nodes_by_valence(find_nodes_and_valence(gt_mask))
    results = {}
    for valence in [1, 2, 3, 4]:
        pred_nodes = pred_nodes_by_valence.get(valence, [])
        gt_nodes = gt_nodes_by_valence.get(valence, [])
        matches, unmatched_pred, unmatched_gt = bipartite_match(pred_nodes, gt_nodes)
        TP = len(matches)
        FP = len(unmatched_pred)
        FN = len(unmatched_gt)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        results[valence] = {'precision': precision, 'recall': recall,
                            'TP': TP, 'FP': FP, 'FN': FN}
    return results

def aggregate_results(all_results):
    # Average precision/recall per valence
    agg = {v: {'precision': [], 'recall': []} for v in [1,2,3,4]}
    for res in all_results:
        for v in [1,2,3,4]:
            agg[v]['precision'].append(res[v]['precision'])
            agg[v]['recall'].append(res[v]['recall'])
    summary = {}
    for v in [1,2,3,4]:
        summary[v] = {
            'precision': np.mean(agg[v]['precision']),
            'recall': np.mean(agg[v]['recall'])
        }
    return summary

if __name__ == "__main__":
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetMultiTask(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.eval()

    # Load test data
    output_dir = '/content/data/thinning/Oxford_split'
    _, _, test_loader = get_dataloaders(output_dir)

    all_results = []
    with torch.no_grad():
        for images, skel_masks, _ in test_loader:
            images = images.to(device)
            pred_skel, _ = model(images)
            pred_skel_bin = (torch.sigmoid(pred_skel) > 0.5).cpu().numpy()
            skel_masks = skel_masks.cpu().numpy()
            for i in range(images.size(0)):
                result = evaluate_node_metrics(pred_skel_bin[i,0], skel_masks[i,0])
                all_results.append(result)

    summary = aggregate_results(all_results)
    print("\nNode Precision & Recall by Valence:")
    print("Valence | Precision | Recall")
    print("-----------------------------")
    for v in [1,2,3,4]:
        print(f"   {v}    |  {summary[v]['precision']:.3f}   |  {summary[v]['recall']:.3f}")