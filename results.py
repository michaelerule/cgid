
import pickle
import statsmodels
from neurotools.tools import memoize

@memoize
def get_glm_phase_tuning_cache():
    all_GLM_results = pickle.load(open('all_GLM_results_blockshuffle.p','rb'))
    # There was a mistake and there are some spurious keys that need 
    # to be removed
    for k in all_GLM_results.keys():
        if len(k)<4: del all_GLM_results[k]
    return all_GLM_results

def get_glm_phase_tuning_result(session,area,unit,epoch):
    all_GLM_results = get_glm_phase_tuning_cache()
    return all_GLM_results[session,area,unit,epoch]

    
@memoize
def get_ppc_phase_tuning_cache():
    all_PPC_results = pickle.load(open('all_PPC_results.p','rb'))
    # There was a mistake and there are some spurious keys that need
    #  to be removed
    for k in all_PPC_results.keys():
        if len(k)<4: del all_PPC_results[k]
    return all_PPC_results

def get_ppc_phase_tuning_result(session,area,unit,epoch):
    all_PPC_results = get_ppc_phase_tuning_cache()
    return all_PPC_results[session,area,unit,epoch]

@memoize
def get_high_low_beta_ppc_cache():
    try:
        high_low_beta_ppc_cache
    except:
        high_low_beta_ppc_cache = {}
    try:
        high_low_beta_ppc_cache.update(pickle.load(open('high_low_beta_ppc_cache.p','rb')))
    except:
        print 'no cached PPC results found here'
    return high_low_beta_ppc_cache

def get_high_low_beta_ppc_result(s,a,u,e):
    high_low_beta_ppc_cache = get_high_low_beta_ppc_cache()
    freqs, high_ppc, low_ppc, betapeak = high_low_beta_ppc_cache[s,a,u,e]
    return freqs, high_ppc, low_ppc, betapeak
        
        
        
        

