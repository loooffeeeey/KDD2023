
# City-wide Origin-Destination Matrix Generation via Graph Denoising Diffusion

This repo contains the codes and data for our submitted KDD'23 research track paper under review.

The models are trained on RTX3090-24G.

## Environment

- python == 3.8.13
- torch == 1.12.1+cu113

## Files

- code # python scripts
  - utils # tool codes
    - metrics.py # all metrics
    - MyLogger.py # A logger for logging experimental information
    - procedure.py # pipeline function
    - tool.py # simple tool functions
  - data_load.py # load data
  - eval.py #
  - main.py # entry
  - model.py # models
  - train.py # training scripts
- data # datasets
- exp # experimental information
  - config # configurations
  - logs # losses and evaluation results
  - results # generations
  - running # 
  - runs # for tensorboard
  - weightes # trained model parameters

## Usage

- The experimental configuration can be adapted in *exp/config/xxx.json*
- In the config file, adjust the ***exp_name*** to record meta information for different experiments.
- In the config file, adjust ***src_cities*** and ***tar_cities*** to select the cities for training and testing. The names of the cities need to be consistent with the names of the subdirectories in the *data* directory
- In *main*.py, modify the selected config file.
- Trained models have been saved in *exp/weights*. Adjust ***exp_name*** to load them.

### training

- set ***topo_train*** to ***1*** means training the topology diffusion model
- set ***flow_train*** to ***1*** means training the flow diffusion model
- set ***T_mode*** to ***INIT*** means training the topology diffusion model from scratch
- set ***F_mode*** to ***INIT*** means training the flow diffusion model from scratch
- Extra setting
  - set ***teacher_force***  to ***1*** means training the flow diffusion models in collaborative mode
  - set ***mem_need*** to check GPU memory, at leat 23000

### testing
- set ***topo_train*** and ***flow_train*** to ***0*** to skip the training process
- set ***T_mode*** and ***F_mode*** to ***LOAD*** to load existing trained models