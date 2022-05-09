from operator import itemgetter
import pickle
import pickle5
def keywithmaxval(d, idx):
    """ a) create a list of the dict's keys and values; 
    b) return the key with the max value"""  
    v = list(d.values())
    k = list(d.keys())
    if idx == 0:
        return k[v.index(max(v,key=itemgetter(idx)))]
    else:
        return k[v.index(min(v,key=itemgetter(idx)))]

def show_case():
    with open('network_params_ac.pickle', 'rb+') as handle:
        results_dict = pickle.load(handle)

    with open('network_params_ac_floyd.pickle', 'rb+') as handle:
        results_dict1 = pickle5.load(handle)

    results_dict.update(results_dict1)
    #print(results_dict)
    #print(results_dict.get)
    max_key = max(results_dict, key=results_dict.get)
    max_perf = keywithmaxval(results_dict, 0)
    max_conv = keywithmaxval(results_dict, 1)
    print(max_key, results_dict[max_key])
    print(max_key, results_dict[max_perf])
    print(max_key, results_dict[max_conv])

    with open('network_params_ac.pickle', 'wb') as handle:
        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('network_params_reinforce.pickle', 'rb+') as handle:
        results_dict = pickle.load(handle)

    with open('network_params_reinforce_floyd.pickle', 'rb+') as handle:
        results_dict1 = pickle5.load(handle)

    results_dict.update(results_dict1)
    #print(results_dict)
    #print(results_dict.get)
    max_key = max(results_dict, key=results_dict.get)
    max_perf = keywithmaxval(results_dict, 0)
    max_conv = keywithmaxval(results_dict, 1)
    print(max_key, results_dict[max_key])
    print(max_key, results_dict[max_perf])
    print(max_key, results_dict[max_conv])

    with open('network_params_reinforce.pickle', 'wb') as handle:
        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

nstep = []