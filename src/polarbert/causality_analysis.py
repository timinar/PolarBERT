"""
Causality Analysis Module for PolarBERT

This module provides tools to analyze whether PolarBERT learned relativistic
causality and the speed of light from IceCube data.

Key physics:
    time_norm = 3e4 ns
    space_norm = 500.0 m
    c = 0.299792458 m/ns (speed of light)
    c_ku = c * time_norm / space_norm = 17.987547 (normalized units)

    Minkowski interval: s² = c_ku² * Δt² - Δx² - Δy² - Δz²
    s² > 0: timelike (causal)
    s² < 0: spacelike (non-causal)
    s² = 0: lightlike (on light cone)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass
import logging
from scipy import stats as scipy_stats

# Physics constants
TIME_NORM = 3e4  # nanoseconds
SPACE_NORM = 500.0  # meters
C_MNS = 0.299792458  # speed of light in m/ns
C_KU = C_MNS * TIME_NORM / SPACE_NORM  # ~17.987547 (normalized units)
C_KU_SQUARED = C_KU ** 2  # ~323.55


@dataclass
class IntervalBins:
    """Interval bins for causal classification."""
    strongly_spacelike: Tuple[float, float] = (-np.inf, -1.0)
    weakly_spacelike: Tuple[float, float] = (-1.0, -0.01)
    lightlike: Tuple[float, float] = (-0.01, 0.01)
    weakly_timelike: Tuple[float, float] = (0.01, 1.0)
    strongly_timelike: Tuple[float, float] = (1.0, np.inf)

    def get_bin_edges(self) -> np.ndarray:
        """Return bin edges as array."""
        return np.array([-np.inf, -1.0, -0.01, 0.01, 1.0, np.inf])

    def get_bin_labels(self) -> List[str]:
        """Return bin labels."""
        return ['strongly_spacelike', 'weakly_spacelike', 'lightlike',
                'weakly_timelike', 'strongly_timelike']


# ============================================================================
# Core interval computation
# ============================================================================

def compute_interval(four_vec1: torch.Tensor, four_vec2: torch.Tensor,
                     c_ku_squared: float = C_KU_SQUARED) -> torch.Tensor:
    """
    Compute Minkowski interval between two 4-vectors.

    Args:
        four_vec1: (t, x, y, z) tensor
        four_vec2: (t, x, y, z) tensor
        c_ku_squared: speed of light squared in normalized units

    Returns:
        Interval s² = c² * Δt² - Δx² - Δy² - Δz²
    """
    dt = four_vec1[0] - four_vec2[0]
    dx = four_vec1[1] - four_vec2[1]
    dy = four_vec1[2] - four_vec2[2]
    dz = four_vec1[3] - four_vec2[3]

    return c_ku_squared * dt**2 - dx**2 - dy**2 - dz**2


def compute_interval_matrix(four_vectors: torch.Tensor,
                            c_ku_squared: float = C_KU_SQUARED) -> torch.Tensor:
    """
    Compute all pairwise Minkowski intervals (vectorized).

    Args:
        four_vectors: tensor of shape (n_pulses, 4) with [t, x, y, z]
        c_ku_squared: speed of light squared in normalized units

    Returns:
        Interval matrix of shape (n_pulses, n_pulses)
    """
    n = len(four_vectors)

    # Extract components
    t = four_vectors[:, 0]  # (n,)
    x = four_vectors[:, 1]  # (n,)
    y = four_vectors[:, 2]  # (n,)
    z = four_vectors[:, 3]  # (n,)

    # Compute pairwise differences using broadcasting
    dt = t.unsqueeze(0) - t.unsqueeze(1)  # (n, n)
    dx = x.unsqueeze(0) - x.unsqueeze(1)  # (n, n)
    dy = y.unsqueeze(0) - y.unsqueeze(1)  # (n, n)
    dz = z.unsqueeze(0) - z.unsqueeze(1)  # (n, n)

    # Compute intervals
    intervals = c_ku_squared * dt**2 - dx**2 - dy**2 - dz**2

    return intervals


def get_four_vectors(batch: Tuple, positions: torch.Tensor, idx: int = 0,
                     only_primary: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract (t, x, y, z) 4-vectors from an event in batch.

    Args:
        batch: Model batch data ((x_dict, lengths), targets)
        positions: DOM position tensor (num_doms+1, 3), normalized
        idx: Event index in batch
        only_primary: If True, filter to non-auxiliary pulses only

    Returns:
        four_vectors: tensor of shape (n_pulses, 4) with [t, x, y, z]
        charges: tensor of shape (n_pulses,) with charge values
        mask_info: dict with masking information
    """
    # Handle different batch formats
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        input_data, _ = batch
        if isinstance(input_data, (list, tuple)) and len(input_data) == 2:
            x, lengths = input_data
        else:
            x, lengths = input_data, None
    else:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        lengths = batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else None

    # Extract features
    if isinstance(x, dict):
        features = x['features'][idx]  # (seq_len, 3): time, charge, aux
        dom_ids = x['dom_id'][idx]     # (seq_len,)
    else:
        features = x[idx]
        dom_ids = batch[0]['dom_id'][idx] if isinstance(batch[0], dict) else None

    # Create padding mask
    not_padded = dom_ids != 0

    # Extract time, charge, auxiliary
    times = features[:, 0]
    charges = features[:, 1]
    auxiliary = features[:, 2]

    # Get xyz positions from DOM IDs
    xyz = positions[dom_ids]  # (seq_len, 3)

    # Apply padding mask
    times = times[not_padded]
    charges = charges[not_padded]
    auxiliary = auxiliary[not_padded]
    xyz = xyz[not_padded]

    # Filter to primary pulses if requested
    if only_primary:
        # aux == -0.5 means primary (non-auxiliary)
        primary = (auxiliary == -0.5)
        times = times[primary]
        charges = charges[primary]
        xyz = xyz[primary]

    # Combine into 4-vectors
    four_vectors = torch.cat([times.unsqueeze(1), xyz], dim=1)

    return four_vectors, charges, {'not_padded': not_padded}


def classify_intervals(intervals: torch.Tensor,
                       bins: Optional[IntervalBins] = None) -> torch.Tensor:
    """
    Classify intervals into causal categories.

    Args:
        intervals: Interval values
        bins: IntervalBins object defining bin edges

    Returns:
        Bin indices (0-4) for each interval
    """
    if bins is None:
        bins = IntervalBins()

    edges = bins.get_bin_edges()
    intervals_np = intervals.cpu().numpy() if isinstance(intervals, torch.Tensor) else intervals

    # np.digitize returns bin indices (1-indexed, so subtract 1)
    bin_indices = np.digitize(intervals_np, edges[1:-1])  # Use internal edges

    return torch.tensor(bin_indices)


# ============================================================================
# Experiment 2: Prediction accuracy by interval bin
# ============================================================================

def analyze_prediction_accuracy_by_interval(
    model: nn.Module,
    batch: Tuple,
    positions: torch.Tensor,
    logits: torch.Tensor,
    mask: torch.Tensor,
    c_ku_squared: float = C_KU_SQUARED,
    bins: Optional[IntervalBins] = None
) -> Dict[str, Any]:
    """
    Analyze prediction accuracy stratified by interval bin for a single batch.

    Args:
        model: PolarBERT model (not used directly, for interface consistency)
        batch: Batch data
        positions: DOM positions tensor
        logits: Model output logits (batch_size, seq_len, vocab_size)
        mask: Mask tensor indicating which positions were masked
        c_ku_squared: Speed of light squared
        bins: Interval bin definitions

    Returns:
        Dictionary with accuracy statistics per bin
    """
    if bins is None:
        bins = IntervalBins()

    batch_size = logits.shape[0]
    device = logits.device

    # Results storage
    results_by_bin = {label: {'correct': 0, 'total': 0, 'top5_correct': 0}
                      for label in bins.get_bin_labels()}

    # Process each event
    for event_idx in range(batch_size):
        # Get 4-vectors
        four_vectors, charges, _ = get_four_vectors(batch, positions, event_idx, only_primary=False)

        if len(four_vectors) < 2:
            continue

        # Get event mask (which positions are masked)
        event_mask = mask[event_idx]

        # Find valid positions (not padded)
        if isinstance(batch, (list, tuple)):
            input_data = batch[0] if not isinstance(batch[0], (list, tuple)) else batch[0][0]
            if isinstance(input_data, dict):
                dom_ids = input_data['dom_id'][event_idx]
            else:
                continue
        else:
            continue

        not_padded = dom_ids != 0

        # Apply mask within valid positions
        event_mask_valid = event_mask[not_padded]

        if event_mask_valid.sum() == 0:
            continue

        # Find reference pulse (largest charge among unmasked, primary)
        features = batch[0][0]['features'][event_idx] if isinstance(batch[0], (list, tuple)) else batch[0]['features'][event_idx]
        aux = features[:, 2][not_padded]
        charges_valid = features[:, 1][not_padded]

        is_primary = (aux == -0.5)
        is_unmasked = ~event_mask_valid

        unmasked_primary = is_primary & is_unmasked
        if unmasked_primary.sum() == 0:
            continue

        # Get reference pulse (highest charge among unmasked primary)
        ref_charges = charges_valid.clone()
        ref_charges[~unmasked_primary] = -float('inf')
        ref_idx = ref_charges.argmax()
        ref_4vec = four_vectors[ref_idx]

        # Get masked pulse indices
        masked_indices = torch.where(event_mask_valid)[0]

        # For each masked pulse, compute interval to reference and check prediction
        for masked_idx in masked_indices:
            masked_4vec = four_vectors[masked_idx]

            # Compute interval
            interval = compute_interval(ref_4vec, masked_4vec, c_ku_squared)

            # Classify interval
            bin_idx = classify_intervals(interval.unsqueeze(0), bins)[0].item()
            bin_label = bins.get_bin_labels()[bin_idx]

            # Get true DOM ID
            true_dom = dom_ids[not_padded][masked_idx].item()

            # Get prediction (need to map back to original sequence position)
            # Find the original position of this masked pulse
            orig_positions = torch.where(not_padded)[0]
            orig_idx = orig_positions[masked_idx].item()

            pred_logits = logits[event_idx, orig_idx]
            pred_dom = pred_logits.argmax().item()

            # Top-5 prediction
            top5_preds = pred_logits.topk(5).indices.tolist()

            # Update statistics
            results_by_bin[bin_label]['total'] += 1
            if pred_dom == true_dom:
                results_by_bin[bin_label]['correct'] += 1
            if true_dom in top5_preds:
                results_by_bin[bin_label]['top5_correct'] += 1

    # Compute accuracies
    for label in bins.get_bin_labels():
        total = results_by_bin[label]['total']
        if total > 0:
            results_by_bin[label]['accuracy'] = results_by_bin[label]['correct'] / total
            results_by_bin[label]['top5_accuracy'] = results_by_bin[label]['top5_correct'] / total
        else:
            results_by_bin[label]['accuracy'] = np.nan
            results_by_bin[label]['top5_accuracy'] = np.nan

    return results_by_bin


def aggregate_accuracy_results(results_list: List[Dict]) -> pd.DataFrame:
    """
    Aggregate accuracy results from multiple batches.

    Args:
        results_list: List of results dictionaries from analyze_prediction_accuracy_by_interval

    Returns:
        DataFrame with aggregated statistics
    """
    bins = IntervalBins()
    aggregated = {label: {'correct': 0, 'total': 0, 'top5_correct': 0}
                  for label in bins.get_bin_labels()}

    for result in results_list:
        for label in bins.get_bin_labels():
            aggregated[label]['correct'] += result[label]['correct']
            aggregated[label]['total'] += result[label]['total']
            aggregated[label]['top5_correct'] += result[label]['top5_correct']

    # Create DataFrame
    rows = []
    for label in bins.get_bin_labels():
        total = aggregated[label]['total']
        if total > 0:
            acc = aggregated[label]['correct'] / total
            top5_acc = aggregated[label]['top5_correct'] / total
            # Wilson score interval for confidence
            ci_low, ci_high = _wilson_score_interval(aggregated[label]['correct'], total)
        else:
            acc = np.nan
            top5_acc = np.nan
            ci_low, ci_high = np.nan, np.nan

        rows.append({
            'interval_bin': label,
            'total': total,
            'correct': aggregated[label]['correct'],
            'accuracy': acc,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'top5_accuracy': top5_acc
        })

    return pd.DataFrame(rows)


def _wilson_score_interval(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute Wilson score confidence interval for a proportion."""
    if n == 0:
        return (np.nan, np.nan)

    z = scipy_stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / n

    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denominator

    return (max(0, center - margin), min(1, center + margin))


# ============================================================================
# Experiment 4: Time scaling analysis
# ============================================================================

def scale_batch_times(batch: Tuple, alpha: float) -> Tuple:
    """
    Scale all times in batch by factor alpha.

    This simulates what the data would look like if the speed of light were different.
    alpha = 1.0: original (correct c)
    alpha < 1.0: faster than light (contracts time)
    alpha > 1.0: slower than light (expands time)

    Args:
        batch: Batch data ((x_dict, lengths), targets)
        alpha: Time scaling factor

    Returns:
        New batch with scaled times
    """
    import copy
    batch = copy.deepcopy(batch)

    # Get features
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        input_data, targets = batch
        if isinstance(input_data, (list, tuple)):
            x, lengths = input_data
        else:
            x = input_data
            lengths = None
    else:
        raise ValueError("Unexpected batch format")

    if isinstance(x, dict):
        # Scale time (first feature)
        x['features'][:, :, 0] = x['features'][:, :, 0] * alpha

        # Reconstruct batch
        if lengths is not None:
            return ([x, lengths], targets)
        else:
            return (x, targets)
    else:
        raise ValueError("Expected x to be a dict with 'features' key")


def compute_loss_for_batch(model: nn.Module, batch: Tuple, device: str = 'cpu') -> float:
    """
    Compute masked prediction loss for a batch.

    Args:
        model: PolarBERT model
        batch: Batch data
        device: Device to run on

    Returns:
        Loss value
    """
    from polarbert.analysis_utils import batch_to_device

    model.eval()
    batch_device, _ = batch_to_device(batch, device)

    with torch.no_grad():
        logits, mask, charge_hat, padding_mask = model(batch_device)

        if mask is None or mask.sum() == 0:
            return np.nan

        # Get target DOM IDs
        target_dom_ids = batch_device[0]['dom_id']

        # Apply mask and padding mask
        final_mask = mask & ~padding_mask

        # Calculate cross-entropy loss
        loss_unreduced = F.cross_entropy(
            logits.permute(0, 2, 1), target_dom_ids, reduction='none'
        )

        # Average per sample, then across batch
        loss_per_sample = (loss_unreduced * final_mask).sum(dim=1) / (final_mask.sum(dim=1) + 1e-8)
        loss = loss_per_sample.mean()

    return loss.item()


def loss_vs_time_scaling(
    model: nn.Module,
    dataloader,
    alpha_values: List[float],
    n_batches: int = 50,
    device: str = 'cpu'
) -> pd.DataFrame:
    """
    Compute loss for different time scaling factors.

    Args:
        model: PolarBERT model
        dataloader: Data loader
        alpha_values: List of scaling factors to test
        n_batches: Number of batches to average over
        device: Device to run on

    Returns:
        DataFrame with loss values for each alpha
    """
    results = []

    for alpha in alpha_values:
        losses = []

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= n_batches:
                break

            # Scale times
            scaled_batch = scale_batch_times(batch, alpha)

            # Compute loss
            loss = compute_loss_for_batch(model, scaled_batch, device)
            if not np.isnan(loss):
                losses.append(loss)

        if losses:
            mean_loss = np.mean(losses)
            std_loss = np.std(losses)
            se_loss = std_loss / np.sqrt(len(losses))
        else:
            mean_loss = np.nan
            std_loss = np.nan
            se_loss = np.nan

        results.append({
            'alpha': alpha,
            'mean_loss': mean_loss,
            'std_loss': std_loss,
            'se_loss': se_loss,
            'n_batches': len(losses)
        })

        logging.info(f"alpha={alpha:.2f}: loss={mean_loss:.4f} ± {se_loss:.4f}")

    return pd.DataFrame(results)


# ============================================================================
# Experiment 5: Light cone visualization
# ============================================================================

def prepare_light_cone_data(
    batch: Tuple,
    positions: torch.Tensor,
    logits: Optional[torch.Tensor],
    mask: Optional[torch.Tensor],
    event_idx: int = 0,
    c_ku: float = C_KU
) -> pd.DataFrame:
    """
    Prepare data for light cone visualization.

    Args:
        batch: Batch data
        positions: DOM positions tensor
        logits: Model output logits (optional)
        mask: Mask tensor (optional)
        event_idx: Event index in batch
        c_ku: Speed of light in normalized units

    Returns:
        DataFrame with columns: dt, dr, interval, is_timelike, prediction_correct
    """
    # Get 4-vectors
    four_vectors, charges, _ = get_four_vectors(batch, positions, event_idx, only_primary=False)

    if len(four_vectors) < 2:
        return pd.DataFrame()

    # Find reference pulse (largest charge)
    ref_idx = charges.argmax()
    ref_4vec = four_vectors[ref_idx]

    rows = []
    for i, pulse_4vec in enumerate(four_vectors):
        if i == ref_idx:
            continue

        # Time difference
        dt = (pulse_4vec[0] - ref_4vec[0]).item()

        # Spatial distance
        dx = pulse_4vec[1] - ref_4vec[1]
        dy = pulse_4vec[2] - ref_4vec[2]
        dz = pulse_4vec[3] - ref_4vec[3]
        dr = torch.sqrt(dx**2 + dy**2 + dz**2).item()

        # Interval
        interval = compute_interval(ref_4vec, pulse_4vec, C_KU_SQUARED).item()

        rows.append({
            'dt': dt,
            'dr': dr,
            'interval': interval,
            'is_timelike': interval > 0,
            'pulse_idx': i,
            'charge': charges[i].item()
        })

    return pd.DataFrame(rows)


def plot_light_cone(
    data: pd.DataFrame,
    c_ku: float = C_KU,
    ax=None,
    title: str = "Light Cone Structure",
    color_by: str = 'is_timelike'
):
    """
    Create light cone visualization.

    Args:
        data: DataFrame from prepare_light_cone_data
        c_ku: Speed of light in normalized units
        ax: Matplotlib axis (creates new if None)
        title: Plot title
        color_by: Column to use for coloring points

    Returns:
        Matplotlib axis
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Plot light cone
    dr_max = data['dr'].max() * 1.1 if len(data) > 0 else 1.0
    dr_range = np.linspace(0, dr_max, 100)

    # Future light cone (positive time)
    ax.plot(dr_range, dr_range / c_ku, 'r--', linewidth=2, label='Light cone')
    # Past light cone (negative time)
    ax.plot(dr_range, -dr_range / c_ku, 'r--', linewidth=2)

    # Fill regions
    ax.fill_between(dr_range, dr_range / c_ku, dr_max / c_ku * 1.1,
                    alpha=0.1, color='blue', label='Timelike (causal)')
    ax.fill_between(dr_range, -dr_range / c_ku, -dr_max / c_ku * 1.1,
                    alpha=0.1, color='blue')
    ax.fill_between(dr_range, -dr_range / c_ku, dr_range / c_ku,
                    alpha=0.1, color='gray', label='Spacelike (non-causal)')

    # Plot data points
    if len(data) > 0:
        if color_by == 'is_timelike':
            colors = ['blue' if t else 'gray' for t in data['is_timelike']]
        elif color_by in data.columns:
            colors = data[color_by]
        else:
            colors = 'blue'

        scatter = ax.scatter(data['dr'], data['dt'], c=colors,
                            s=50, alpha=0.6, edgecolors='black', linewidths=0.5)

    ax.set_xlabel('Spatial distance Δr (normalized)', fontsize=12)
    ax.set_ylabel('Time difference Δt (normalized)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    return ax


# ============================================================================
# Statistical utilities
# ============================================================================

def bootstrap_ci(data: np.ndarray, statistic=np.mean,
                 n_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.

    Args:
        data: Data array
        statistic: Function to compute statistic
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level

    Returns:
        (point_estimate, ci_low, ci_high)
    """
    n = len(data)
    point_estimate = statistic(data)

    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))

    bootstrap_stats = np.array(bootstrap_stats)
    alpha = 1 - confidence
    ci_low = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return point_estimate, ci_low, ci_high


def permutation_test(x: np.ndarray, y: np.ndarray,
                     n_permutations: int = 10000) -> Tuple[float, float]:
    """
    Permutation test for correlation significance.

    Args:
        x: First variable
        y: Second variable
        n_permutations: Number of permutations

    Returns:
        (observed_correlation, p_value)
    """
    observed_corr = np.corrcoef(x, y)[0, 1]

    count_extreme = 0
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        perm_corr = np.corrcoef(x, y_perm)[0, 1]
        if abs(perm_corr) >= abs(observed_corr):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)

    return observed_corr, p_value


# ============================================================================
# Data interval distribution analysis (Experiment 1)
# ============================================================================

def analyze_interval_distribution(
    dataloader,
    positions: torch.Tensor,
    n_events: int = 10000,
    c_ku_squared: float = C_KU_SQUARED
) -> Dict[str, Any]:
    """
    Analyze the distribution of Minkowski intervals in the dataset.

    Args:
        dataloader: Data loader
        positions: DOM positions tensor
        n_events: Number of events to analyze
        c_ku_squared: Speed of light squared

    Returns:
        Dictionary with interval statistics
    """
    all_intervals = []
    events_processed = 0

    for batch in dataloader:
        if events_processed >= n_events:
            break

        # Get batch size
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            input_data = batch[0]
            if isinstance(input_data, (list, tuple)):
                x = input_data[0]
            else:
                x = input_data
        else:
            continue

        if isinstance(x, dict):
            batch_size = x['features'].shape[0]
        else:
            continue

        # Process each event
        for idx in range(batch_size):
            if events_processed >= n_events:
                break

            four_vectors, _, _ = get_four_vectors(batch, positions, idx, only_primary=True)

            if len(four_vectors) > 1:
                intervals = compute_interval_matrix(four_vectors, c_ku_squared)
                # Get upper triangle (exclude diagonal)
                upper_tri = intervals[torch.triu(torch.ones_like(intervals), diagonal=1) == 1]
                all_intervals.extend(upper_tri.tolist())

            events_processed += 1

    all_intervals = np.array(all_intervals)

    # Compute statistics
    timelike = all_intervals > 0
    spacelike = all_intervals < 0

    results = {
        'n_events': events_processed,
        'n_intervals': len(all_intervals),
        'timelike_fraction': timelike.mean(),
        'spacelike_fraction': spacelike.mean(),
        'mean_interval': all_intervals.mean(),
        'median_interval': np.median(all_intervals),
        'std_interval': all_intervals.std(),
        'mean_timelike': all_intervals[timelike].mean() if timelike.any() else np.nan,
        'mean_spacelike': all_intervals[spacelike].mean() if spacelike.any() else np.nan,
        'all_intervals': all_intervals
    }

    return results


# ============================================================================
# Attention analysis (Experiment 3)
# ============================================================================

class AttentionExtractableTransformer(nn.Module):
    """
    Wrapper that extracts attention weights from FlashTransformer.

    Since Flash Attention doesn't return attention weights by default,
    we need to compute them manually when requested.
    """

    def __init__(self, base_model: nn.Module):
        """
        Args:
            base_model: FlashTransformer or similar model
        """
        super().__init__()
        self.base_model = base_model

    def forward(self, input, return_attention: bool = False):
        """
        Forward pass with optional attention extraction.

        Args:
            input: Model input (x_dict, lengths)
            return_attention: If True, compute and return attention weights

        Returns:
            If return_attention=False: same as base model
            If return_attention=True: (logits, mask, charge, padding_mask, attention_weights)
                where attention_weights is (batch, n_layers, n_heads, seq_len, seq_len)
        """
        if not return_attention:
            return self.base_model(input)

        # Get embeddings
        embeddings, padding_mask, mask = self.base_model.embedding(input)

        # Collect attention weights from each layer
        attention_weights_all_layers = []

        x = embeddings
        for block in self.base_model.transformer_blocks:
            # We need to extract attention from the block
            # This requires modifying how we call attention
            x, attn_weights = self._forward_block_with_attention(block, x, padding_mask)
            attention_weights_all_layers.append(attn_weights)

        # Apply final layer norm if present
        if hasattr(self.base_model, 'use_final_layer_norm') and self.base_model.use_final_layer_norm:
            x = self.base_model.final_layer_norm(x)

        # Get predictions
        sequence_output = x[:, 1:, :]  # Remove CLS token
        logits = self.base_model.unembedding(sequence_output)

        charge_hat = None
        if hasattr(self.base_model, 'charge_prediction') and self.base_model.charge_prediction is not None:
            cls_output = x[:, 0, :]
            charge_hat = self.base_model.charge_prediction(cls_output)

        # Stack attention weights: (batch, n_layers, n_heads, seq, seq)
        attention_weights = torch.stack(attention_weights_all_layers, dim=1)

        return logits, mask, charge_hat, padding_mask[:, 1:], attention_weights

    def _forward_block_with_attention(self, block, x, padding_mask):
        """
        Forward through a transformer block, returning attention weights.
        """
        # Pre-norm
        normed = block.layer_norm1(x)

        # Attention with explicit weight computation
        attn_output, attn_weights = self._attention_with_weights(
            block.attention, normed, padding_mask
        )

        # Residual
        x = x + attn_output

        # FFD
        x = x + block.feed_forward(block.layer_norm2(x))

        return x, attn_weights

    def _attention_with_weights(self, attention_module, x, padding_mask):
        """
        Compute attention with explicit weight calculation.

        Compatible with FlashTransformer's Attention module which uses:
        - n_heads (not num_heads)
        - wq, wk, wv (separate projections, not combined qkv)
        - wo (output projection, not out)
        """
        batch_size, seq_len, embed_dim = x.shape
        n_heads = attention_module.n_heads
        head_dim = attention_module.head_dim

        # Get Q, K, V using separate projections
        xq = attention_module.wq(x)  # (batch, seq, n_heads * head_dim)
        xk = attention_module.wk(x)
        xv = attention_module.wv(x)

        # Reshape: (batch, seq, n_heads, head_dim) -> (batch, n_heads, seq, head_dim)
        xq = xq.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
        xk = xk.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
        xv = xv.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)

        # Compute attention scores
        # Check if muP is enabled for scaling
        if hasattr(attention_module, 'is_mup_enabled') and attention_module.is_mup_enabled:
            scale = 1.0 / head_dim
        else:
            scale = 1.0 / (head_dim ** 0.5)

        attn_scores = torch.matmul(xq, xk.transpose(-2, -1)) * scale  # (batch, heads, seq, seq)

        # Apply padding mask (FlashTransformer uses inverted logic)
        if padding_mask is not None:
            # Expand mask for broadcasting: (batch, seq) -> (batch, 1, 1, seq)
            mask_expanded = padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask_expanded, float('-inf'))

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Handle NaN from all-masked rows
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Apply attention
        attn_output = torch.matmul(attn_weights, xv)  # (batch, heads, seq, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # Output projection
        attn_output = attention_module.wo(attn_output)

        return attn_output, attn_weights


def attention_interval_correlation(
    attention_weights: torch.Tensor,
    interval_matrix: torch.Tensor,
    layer_idx: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute correlation between attention weights and causal structure.

    Args:
        attention_weights: (batch, n_layers, n_heads, seq, seq) or (n_heads, seq, seq)
        interval_matrix: (seq, seq) Minkowski intervals
        layer_idx: If provided, analyze only this layer

    Returns:
        Dictionary with correlation statistics
    """
    # Handle different input shapes
    if attention_weights.dim() == 5:
        # (batch, layers, heads, seq, seq) -> average over batch and heads
        if layer_idx is not None:
            attn = attention_weights[:, layer_idx, :, :, :].mean(dim=(0, 1))
        else:
            attn = attention_weights.mean(dim=(0, 1, 2))
    elif attention_weights.dim() == 4:
        # (layers, heads, seq, seq) -> average over heads
        if layer_idx is not None:
            attn = attention_weights[layer_idx].mean(dim=0)
        else:
            attn = attention_weights.mean(dim=(0, 1))
    else:
        attn = attention_weights.mean(dim=0) if attention_weights.dim() == 3 else attention_weights

    # Flatten (excluding diagonal)
    seq_len = attn.shape[-1]
    mask = ~torch.eye(seq_len, dtype=torch.bool, device=attn.device)

    attn_flat = attn[mask].cpu().numpy()
    interval_flat = interval_matrix[mask[:interval_matrix.shape[0], :interval_matrix.shape[1]]].cpu().numpy()

    # Ensure same length
    min_len = min(len(attn_flat), len(interval_flat))
    attn_flat = attn_flat[:min_len]
    interval_flat = interval_flat[:min_len]

    # Compute correlations
    timelike_indicator = (interval_flat > 0).astype(float)

    pearson_corr, pearson_p = scipy_stats.pearsonr(attn_flat, timelike_indicator)
    spearman_corr, spearman_p = scipy_stats.spearmanr(attn_flat, timelike_indicator)

    return {
        'pearson_correlation': pearson_corr,
        'pearson_pvalue': pearson_p,
        'spearman_correlation': spearman_corr,
        'spearman_pvalue': spearman_p,
        'n_pairs': len(attn_flat)
    }
