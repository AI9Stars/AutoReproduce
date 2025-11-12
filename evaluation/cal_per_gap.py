import numpy as np

def calculate_perf_gap(ref_val, agent_val):
    numerator = abs(ref_val - agent_val)
    denominator = max(ref_val, agent_val)
    if denominator == 0:
        return 0.0
    else:
        return numerator / denominator

if __name__ == "__main__":
    p_reference_dict = {
        "IEBins": 0.8854,
        "iTransformer": 0.4008,
        "DKD": 74.56,
        "SimVP": 26.19,
        "HumanMAC": 0.2195,
        "SFNet": 39.68,
        "LSM": 0.0074,
        "Swin-Unet": 0.7309,
        "TDGNN-w": 0.7651,
        "TimeVAE": 0.2083,
        "WCDM": 0.7938,
        "BSPM": 0.1921,
        "DAT-S": 38.48
    }
    ### update agent reproduced performance
    p_agent_dict = {}
    individual_gaps_list = []
    n = len(p_reference_dict)
    for method_name, ref_value in p_reference_dict.items():
        if method_name in p_agent_dict:
            agent_value = p_agent_dict[method_name]
            gap = calculate_perf_gap(ref_value, agent_value)
            individual_gaps_list.append(gap)
        else:
            individual_gaps_list.append(1.0)
    total_performance_gap = np.mean(individual_gaps_list)
    print(f"\nFinal Performance Gap: {total_performance_gap:.6f}")
