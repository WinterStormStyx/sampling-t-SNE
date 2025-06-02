import visualization_helper as vh
from utils.utils import DataType

if __name__ == "__main__":
    #mnist = [1/7000, 7/7000, 21/7000, 144/7000]
    #celegans = [3/8970, 12/8970, 34/8970, 105/8970]
    #wong = [3/32745, 30/32745, 100/32745, 200/32745]

    for perp in [3/8970, 12/8970, 34/8970, 105/8970]:
        for sampling_rate in [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]:
            vh.comparison_pipeline(DataType.CELEGANS, perp, sampling_rate, seed=42)
