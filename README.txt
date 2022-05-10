To get info of the possible arguments run: python Experiments.py
To tune the parameters run: python Experiments.py --method <METHOD> -tune <Network settings to tune (int)> **kwargs
To compare AC with REINFORCE run: python Experiments.py --method <any> -comparison 1 **kwargs
To run the ablation run: python Experiments.py --method <METHOD> -ablation 1 **kwargs
If no optional arguments are given, the best settings of the previously performed tuning are used.