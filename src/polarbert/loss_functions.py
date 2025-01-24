import torch


def angular_dist_score(az_true, zen_true, az_pred, zen_pred):
    '''
    PyTorch version of the loss function `angular_dist_score`.
    '''
    
    assert(torch.all(torch.isfinite( az_pred))) # DEBUG
    assert(torch.all(torch.isfinite(zen_pred))) # DEBUG
    
    # pre-compute all sine and cosine values
    sa1 = torch.sin(az_true)
    ca1 = torch.cos(az_true)
    sz1 = torch.sin(zen_true)
    cz1 = torch.cos(zen_true)
    
    sa2 = torch.sin(az_pred)
    ca2 = torch.cos(az_pred)
    sz2 = torch.sin(zen_pred)
    cz2 = torch.cos(zen_pred)
    
    # scalar product of the two cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1*sz2*(ca1*ca2 + sa1*sa2) + (cz1*cz2)
    
    # scalar product of two unit vectors is always between -1 and 1, this is against nummerical instability
    # that might otherwise occure from the finite precision of the sine and cosine functions
    scalar_prod =  torch.clip(scalar_prod, -1, 1)
    
    # convert back to an angle (in radian)
    return torch.mean(torch.abs(torch.arccos(scalar_prod)))

def angles_to_unit_vector(azimuth, zenith):
    return torch.stack([
        torch.cos(azimuth) * torch.sin(zenith),
        torch.sin(azimuth) * torch.sin(zenith),
        torch.cos(zenith)
    ], dim=1)

def unit_vector_to_angles(n, check_unit_norm=False):
    norm = torch.sqrt(torch.sum(n**2, dim=1))
    if check_unit_norm:
        assert(torch.allclose(norm, torch.full_like(norm, 1.)))
    x = n[:,0]
    y = n[:,1]
    z = n[:,2]
    azimuth = torch.atan2(y, x)
    zenith = torch.arccos(z / norm)
    return torch.stack([azimuth, zenith], dim=1)

def angular_dist_score_unit_vectors(n_true, n_pred, epsilon=0):
    assert(torch.all(torch.isfinite(n_true))) # DEBUG
    assert(torch.all(torch.isfinite(n_pred))) # DEBUG
    scalar_prod = torch.sum(n_true * n_pred, dim=1)
    scalar_prod = torch.clip(scalar_prod, -1+epsilon, 1-epsilon)
    return torch.mean(torch.abs(torch.arccos(scalar_prod)))