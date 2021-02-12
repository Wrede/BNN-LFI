from bcnn_model import Classifier, inBin, sample_local, sample_local_2D, sample_local_2D_adaptive, exp_adaptive_thresh
from ma2_model import newData, triangle_prior, uniform_prior
import numpy as np
import time
import os

def run_bnn(total_runs=10, num_generation=6, ID='test', num_bins=3, seed=None, use_thresh=True, adaptive=False, thresh=0.05):
    np.random.seed(seed)
    save_folder = ID
    Ndata = 3000 # number of model samples
    num_monte_carlo = 1000 #number of monte carlo samples of predictive posterior
    batch_size = 256 #batch size in training NN
    #thresh = 0.05 # explotation on resampling, apply threshhold on categorical posterior distribution.
    if adaptive:
        adaptive_thresh = exp_adaptive_thresh(0.5, 5, num_generation)
    use_seed = None
    multi_dim = True
    use_small = True
    use_local_tresh = use_thresh
    num_bins_per_param = num_bins
    initial_prior = uniform_prior 
    #initial_prior = triangle_prior 
    result_posterior = [] #store posterior samples per round
    store_time = [] 

    target_ts = np.load('target_ts.npy') #Load the observation

    for run in range(total_runs):
        print(f'starting run {run}')
        theta = []
        theta_corrected = []
        time_ticks = []
        theta.append(initial_prior(Ndata))

        for i in range(num_generation):
            time_begin = time.time()
            print(f'starting generation {i}')

            # generate new data
            data_ts, data_thetas = newData(theta[i])

            # of the previous gen dataset, which ones can we re-use? Goal is to maximize 
            # the number of datapoints available.
            if i > 0:
                data_ts_, data_thetas_ = inBin(data_ts, data_thetas, theta[i])
                data_ts = np.append(data_ts, data_ts_, axis=0)
                data_thetas = np.append(data_thetas, data_thetas_, axis=0)

            # saving not only the full parameter arrays THETA but also the ones that are 
            # removed because of timeout signal. 
            theta_corrected.append(data_thetas)

            # Classify new data
            lv_c = Classifier(name_id=f'ma2_{i}', seed=use_seed)
            print('trainDataSize: {}'.format(len(data_thetas)))

            lv_c.train_thetas = data_thetas
            lv_c.train_ts = data_ts

            lv_c.run(target=target_ts,num_bins=num_bins_per_param, num_monte_carlo=num_monte_carlo, batch_size=batch_size, 
                     split=True, verbose=False, multi_dim=multi_dim, use_small=use_small)

            # save model for evaluation
            #if multi_dim:
            #    lv_c.model1.save(f'{save_folder}/ma2_run{run}_gen{i}_model_multidim_{ID}')
            #else:
            #    lv_c.model1.save(f'{save_folder}/ma2_run{run}_gen{i}_model1_{ID}')
            #    lv_c.model2.save(f'{save_folder}/ma2_run{run}_gen{i}_model2_{ID}')


            # resample
            if multi_dim:
                if adaptive:
                    new_samples, new_bins = sample_local_2D_adaptive(lv_c.probs1, lv_c.multidim_bins, num_samples=Ndata, use_thresh=use_local_tresh, thresh=adaptive_thresh[i])
                    theta.append(new_samples)
                else:
                    new_samples, new_bins = sample_local_2D(lv_c.probs1, lv_c.multidim_bins, num_samples=Ndata, use_thresh=use_local_tresh, thresh=thresh)
                    theta.append(new_samples)

            else: 
                new_rate1, new_bins1 = sample_local(lv_c.probs1, lv_c.bins_rate1, num_samples=Ndata, use_thresh=use_local_tresh, thresh=thresh)
                new_rate2, new_bins2 = sample_local(lv_c.probs2, lv_c.bins_rate2, num_samples=Ndata, use_thresh=use_local_tresh, thresh=thresh)

                theta.append(np.vstack((new_rate1,new_rate2)).T)
            time_ticks.append(time.time() - time_begin)
        
        result_posterior.append(theta)
        store_time.append(time_ticks)
    return np.asarray(result_posterior), np.asarray(store_time)

def bnn_experiment():
    ID = 'data'
    try:
        os.mkdir(ID)
    except FileExistsError:
        print(f'{ID} folder already exists, continue...')

    #exponential threshold
    bcnn_post, bcnn_time = run_bnn(total_runs=10, num_generation=6, seed=3, ID=ID, num_bins=4, adaptive=True, use_thresh=True)
    np.save(f'{ID}/bcnn_{ID}_exp_thresh_post', bcnn_post)
        
    bcnn_post, bcnn_time = run_bnn(total_runs=10, num_generation=6, seed=3, ID=ID, num_bins=4, use_thresh=True)
    np.save(f'{ID}/bcnn_{ID}_post', bcnn_post)
    np.save(f'{ID}/bcnn_{ID}_time', bcnn_time)


    #Bins experiemnt
    bcnn_post, bcnn_time = run_bnn(total_runs=10, num_generation=6, seed=3, ID=ID, num_bins=3, use_thresh=True)
    np.save(f'{ID}/bcnn_{ID}_bins3_post', bcnn_post)
    bcnn_post, bcnn_time = run_bnn(total_runs=10, num_generation=6, seed=3, ID=ID, num_bins=5, use_thresh=True)
    np.save(f'{ID}/bcnn_{ID}_bins5_post', bcnn_post)

    #Threshold experiemnt
    bcnn_post, bcnn_time = run_bnn(total_runs=10, num_generation=6, seed=3, ID=ID, num_bins=4, use_thresh=False)
    np.save(f'{ID}/bcnn_{ID}_no_thresh_post', bcnn_post)




        


