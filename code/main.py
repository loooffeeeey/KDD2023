import datetime
import json
import os

import setproctitle
from torch.utils.data import DataLoader
from train import *
from utils.procedure import *
from utils.tool import map_location_dict
from data_load import *
from model import *
from utils.MyLogger import Logger
import  socket


def main(config):
    # load data
    data = prepare_data(config)

    # set input dim for model
    config["GAT_indim"] = data["src"][0]["attr"].shape[1]

    # logger
    logger = Logger(config)
    if config["T_mode"] == "LOAD" or config["F_mode"] == "LOAD":
        logger.load_exp_log()
        logger.load_exp_tensorboard()

    # model
    model_T = TopoDiffusion(config)
    model_F = FlowDiffusion(config)
    
    if config["topo_diffusion"] == 1:
        if config["T_mode"] == "LOAD":
            # topo model
            if config["T_load"] == "last":
                topo_path = logger.model_path[:-4] + "_topo.pkl"
            elif config["T_load"] == "best":
                topo_path = logger.model_path[:-4] + "_topo_best.pkl"
            if "device" in config.keys():
                state_dict_topo = torch.load(topo_path, map_location=config["device"])
                model_T.load_state_dict(state_dict_topo)
            else:
                state_dict_topo = torch.load(topo_path)
                map_locations = map_location_dict(state_dict=state_dict_topo, config=config)
                model_T.load_state_dict(torch.load(topo_path, map_location=map_locations))
    if config["flow_diffusion"] == 1:
        if config["F_mode"] == "LOAD":
            # flow model
            if config["F_load"] == "last":
                flow_path = logger.model_path[:-4] + "_flow.pkl"
            elif config["F_load"] == "best":
                flow_path = logger.model_path[:-4] + "_flow_best.pkl"
            if "device" in config.keys():
                state_dict_flow = torch.load(flow_path, map_location=config["device"])
                model_F.load_state_dict(state_dict_flow)
            else:
                state_dict_flow = torch.load(flow_path)
                map_locations = map_location_dict(state_dict=state_dict_flow, config=config)
                model_F.load_state_dict(torch.load(flow_path, map_location=map_locations))

    # train
    train(config, model_T, model_F, data, logger)

    

if __name__ == "__main__":

    config = get_config("exp/config/Cook_NYC.json")
    # random seed
    setRandomSeed(config["random_seed"])

    main(config)