
out_l_cent = 50.0
out_l_norm = 100.0
out_ab_norm = 110.0

def normalize_l( in_l):
    return (in_l - out_l_cent) / out_l_norm

def denormalize_l(in_l):
    return in_l * out_l_norm + out_l_cent

def normalize_ab(in_ab):
    return in_ab / out_ab_norm

def denormalize_ab(in_ab):
    return in_ab * out_ab_norm
    
