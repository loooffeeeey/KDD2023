import os

import numpy as np

import torch

from copy import deepcopy
from random import shuffle
from pprint import pprint

from utils.tool import ToThTensor, rescale_od, map_location_dict, permute_node_order
from utils.metrics import *

# from eval import check_condition

from tqdm import tqdm

import torch.nn.functional as F


def train(config, model_T, model_F, data, logger):

    if config["topo_diffusion"] == 1:

        if config["topo_train"] == 1:
            #### topo 生成部分
            optm_T = torch.optim.AdamW(model_T.NN.parameters(), lr=config["lr_topo"])

            if config["T_mode"] == "LOAD":
                if "device" in config.keys():
                    state_dict_topo = torch.load(logger.optimizer_path[:-4] + "_topo.pkl", map_location=config["device"])
                    optm_T.load_state_dict(state_dict_topo)
                else:
                    state_dict_topo = torch.load(logger.optimizer_path[:-4] + "_topo.pkl")
                    map_locations = map_location_dict(state_dict=state_dict_topo, config=config)
                    optm_T.load_state_dict(torch.load(logger.optimizer_path[:-4] + "_topo.pkl", map_location=map_locations))

            if config["T_mode"] == "INIT":
                start_epochs = 0
            elif config["T_mode"] == "LOAD":
                start_epochs = logger.exp_log["train_log"]["topo_epochs"]

            for epoch in range(start_epochs, config["EPOCH"]):
                for i in range(len(data["src"])):
                    # source city
                    attr_src = ToThTensor(data["src"][i]["attr"])
                    dis_src = ToThTensor(data["src"][i]["dis"])
                    g_src = data["src"][i]["g"]
                    od_src = ToThTensor(data["src"][i]["od"])
                    od_topo_src = ToThTensor(data["src"][i]["od_topo"])

                    # 尝试去打乱节点排序顺序
                    attr_src, dis_src, g_src, od_src, od_topo_src = permute_node_order(config, attr_src, dis_src, g_src, od_src, od_topo_src)

                    condition_src = (attr_src, dis_src, g_src, od_topo_src)

                    clean_topo_src = od_topo_src
                    if config["topo_softlabel"] != 1:
                        clean_topo_src = F.one_hot(clean_topo_src.long(), num_classes=config["Topo_e_classes"]).float()

                    print("*" * len("*******Validation*******"), config["exp_name"] + " Epoch.", epoch, "\n Train with", config["src_cities"][i])
                    Ts = [x for x in range(config["T_topo"])]
                    shuffle(Ts)
                    Ts = Ts[:config["topo_num_t_epoch"]]
                    for t in tqdm(Ts):
                        optm_T.zero_grad()
                        t = torch.LongTensor([t])

                        if config["topo_train_objective"] == "x_t_minus_1":
                            loss = model_T.loss_x_t_minus_1(clean_topo_src, t, condition_src)
                        elif config["topo_train_objective"] == "x_0":
                            loss = model_T.loss(clean_topo_src, t, condition_src)

                        loss_value = loss.item()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model_T.parameters(), 1.)
                        optm_T.step()

                        logger.once_record_topo(loss_value, model_T, optm_T, epoch=epoch, t=t.item())
                    logger.upgrade_topo_epochs()
                    logger.save_exp_log()
                
                print("-" * 50)
                    
                    

                if epoch % 10 == 0:
                    # test
                    for i in range(len(data["tar"])):
                        # target city
                        attr_tar = ToThTensor(data["tar"][i]["attr"])
                        dis_tar = ToThTensor(data["tar"][i]["dis"])
                        g_tar = data["tar"][i]["g"]
                        od_tar = ToThTensor(data["tar"][i]["od"])
                        od_topo_tar = ToThTensor(data["tar"][i]["od_topo"])

                        x_0_tar = od_tar.to(config["devices"][0])
                        valid_x_0_tar = rescale_od(x_0_tar.to(torch.device("cpu")), data["tar"][i]["od_min_max"])

                        condition_tar = (attr_tar, dis_tar, g_tar, od_topo_tar)

                        print("Test with", config["tar_cities"][i], ".")
                        with torch.no_grad():
                            # 采样生成
                            x_0_hat_tar = model_T.sample(x_0_tar.shape, condition_tar)
                            logger.save_generation(config["tar_cities"][i], x_0_hat_tar, "topo")
                            all_metrics = cal_all_metrics_topo(x_0_hat_tar.to(od_topo_tar.device), od_topo_tar)
                            logger.summary_all_test_metrics_topo(all_metrics, epoch, config["tar_cities"][i])

                            del x_0_hat_tar
                    if all_metrics["CPC_topo"] > logger.best_test_cpc_topo:
                        logger.best_test_cpc_topo = all_metrics["CPC_topo"]
                        logger.save_model_optm_scheduler(model_T, optm_T, exp="topo_best")
        else:
            # test
            epoch_test = 0
            x_0_hats = []
            for i in range(len(data["tar"])):
                # target city
                attr_tar = ToThTensor(data["tar"][i]["attr"])
                dis_tar = ToThTensor(data["tar"][i]["dis"])
                g_tar = data["tar"][i]["g"]
                od_tar = ToThTensor(data["tar"][i]["od"])
                od_topo_tar = ToThTensor(data["tar"][i]["od_topo"])

                x_0_tar = od_tar.to(config["devices"][0])
                valid_x_0_tar = rescale_od(x_0_tar.to(torch.device("cpu")), data["tar"][i]["od_min_max"])

                condition_tar = (attr_tar, dis_tar, g_tar, od_topo_tar)

                print("Test with", config["tar_cities"][i], ".")
                with torch.no_grad():
                    # 采样生成
                    x_0_hat_tar = model_T.sample(x_0_tar.shape, condition_tar)
                    logger.save_generation(config["tar_cities"][i], x_0_hat_tar, "topo")
                    all_metrics = cal_all_metrics_topo(x_0_hat_tar.to(od_topo_tar.device), od_topo_tar)
                    pprint(all_metrics)
                x_0_hats.append(x_0_hat_tar)
            epoch_test += 1
            print("-" * 50)


    if config["flow_diffusion"]:

        if config["flow_train"] == 1:
            optm_F = torch.optim.AdamW(model_F.NN.parameters(), lr=config["lr_flow"])
            if config["F_mode"] == "LOAD":
                if "device" in config.keys():
                    state_dict_topo = torch.load(logger.optimizer_path[:-4] + "_flow.pkl", map_location=config["device"])
                    optm_F.load_state_dict(state_dict_topo)
                else:
                    state_dict_topo = torch.load(logger.optimizer_path[:-4] + "_flow.pkl")
                    map_locations = map_location_dict(state_dict=state_dict_topo, config=config)
                    optm_F.load_state_dict(torch.load(logger.optimizer_path[:-4] + "_flow.pkl", map_location=map_locations))

            #### flow 生成部分
            for epoch in range(config["EPOCH"]):
                for i in range(len(data["src"])):
                    # source city
                    attr_src = ToThTensor(data["src"][i]["attr"])
                    dis_src = ToThTensor(data["src"][i]["dis"])
                    g_src = data["src"][i]["g"]
                    od_src = ToThTensor(data["src"][i]["od"])
                    od_topo_src = ToThTensor(data["src"][i]["od_topo"])

                    if   (config["topo_diffusion"] == 1) and (config["teacher_force"] == 1):
                        topo = torch.zeros_like(od_topo_src).to(x_0_hats[0].device)
                        topo[(od_topo_src.to(x_0_hats[0].device) == 1) | (x_0_hats[0] == 1)] = 1
                        od_topo_src = topo
                    elif (config["topo_diffusion"] == 1) and (config["teacher_force"] == 0):
                        topo = x_0_hats[0]
                    elif (config["topo_diffusion"] == 0) and (config["teacher_force"] == 0):
                        topo = torch.ones_like(od_topo_src)
                        od_topo_src = topo
                    elif (config["topo_diffusion"] == 0) and (config["teacher_force"] == 1): # 直接使用真值训练
                        pass

                    x_0_src = od_src.to(config["devices"][0])

                    condition_src = (attr_src, dis_src, g_src, od_topo_src)

                    # training
                    print("*" * len("*******Validation*******"), config["exp_name"] + " Epoch.", epoch, "\n Train with", config["src_cities"][i])
                    Ts = [x for x in range(config["T_flow"])]
                    shuffle(Ts)
                    Ts = Ts[:config["flow_num_t_epoch"]]
                    for t in tqdm(Ts):
                        optm_F.zero_grad()
                        t = torch.LongTensor([t])
                        loss = model_F.loss(x_0_src, t, condition_src)
                        loss_value = loss.item()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model_F.parameters(), 1.)
                        optm_F.step()

                        logger.once_record_flow(loss_value, model_F, optm_F, epoch=epoch, t=t.item())
                    logger.upgrade_flow_epochs()
                    logger.save_exp_log()
                    
                print("-" * 50)  

                # valid
                with torch.no_grad():
                    for i in range(len(data["tar"])):
                        # target city
                        attr_tar = ToThTensor(data["tar"][i]["attr"])
                        dis_tar = ToThTensor(data["tar"][i]["dis"])
                        g_tar = data["tar"][i]["g"]
                        od_tar = ToThTensor(data["tar"][i]["od"])
                        od_topo_tar = ToThTensor(data["tar"][i]["od_topo"])

                        if   (config["topo_diffusion"] == 1) and (config["teacher_force"] == 1):
                            od_topo_tar = x_0_hats[i]
                        elif (config["topo_diffusion"] == 1) and (config["teacher_force"] == 0):
                            od_topo_tar = x_0_hats[i]
                        elif (config["topo_diffusion"] == 0) and (config["teacher_force"] == 0):
                            od_topo_tar = torch.ones_like(od_topo_tar)
                        elif (config["topo_diffusion"] == 0) and (config["teacher_force"] == 1): # 直接使用真值训练
                            pass

                        x_0_tar = od_tar.to(config["devices"][0])
                        valid_x_0_tar = rescale_od(x_0_tar.to(torch.device("cpu")), data["tar"][i]["od_min_max"])

                        condition_tar = (attr_tar, dis_tar, g_tar, od_topo_tar)

                        print("Test with", config["tar_cities"][i], ".")
                        
                        if config["sample_method"] == "DDPM":
                            x_seq = model_F.p_sample_loop(x_0_tar.shape, condition_tar)
                        elif config["sample_method"] == "DDIM":
                            x_seq = model_F.DDIM_sample_loop(x_0_tar.shape, condition_tar)

                        x_0_hat = rescale_od(x_seq[-1], data["tar"][i]["od_min_max"])
                        topo = condition_tar[3]
                        zero_flow_idx = topo == 0
                        x_0_hat[zero_flow_idx] = -1 # Zero flows are normalized to -1.
                        x_0_hat = torch.clip(x_0_hat, min=0)
                        logger.save_generation(config["tar_cities"][i], x_0_hat, "flow")

                        x_0_hat, valid_x_0_tar = x_0_hat.round(), valid_x_0_tar.round()
                        all_metrics = cal_all_metrics_flow(x_0_hat, valid_x_0_tar)
                        logger.summary_all_test_metrics(all_metrics, epoch, config["tar_cities"][i])
                    if all_metrics["CPC"] > logger.best_test_cpc_flow:
                        logger.best_test_cpc_flow = all_metrics["CPC"]
                        logger.save_model_optm_scheduler(model_F, optm_F, exp="flow_best")
                print("-" * 50)

        else:    

            # test
            epoch_test = 0
            for i in range(len(data["tar"])):
                # target city
                attr_tar = ToThTensor(data["tar"][i]["attr"])
                dis_tar = ToThTensor(data["tar"][i]["dis"])
                g_tar = data["tar"][i]["g"]
                od_tar = ToThTensor(data["tar"][i]["od"])
                od_topo_tar = ToThTensor(data["tar"][i]["od_topo"])

                if   (config["topo_diffusion"] == 1) and (config["teacher_force"] == 1):
                    od_topo_tar = x_0_hats[i]
                elif (config["topo_diffusion"] == 1) and (config["teacher_force"] == 0):
                    od_topo_tar = x_0_hats[i]
                elif (config["topo_diffusion"] == 0) and (config["teacher_force"] == 0):
                    od_topo_tar = torch.ones_like(od_topo_tar)
                elif (config["topo_diffusion"] == 0) and (config["teacher_force"] == 1): # 直接使用真值训练
                    pass

                x_0_tar = od_tar.to(config["devices"][0])
                valid_x_0_tar = rescale_od(x_0_tar.to(torch.device("cpu")), data["tar"][i]["od_min_max"])

                condition_tar = (attr_tar, dis_tar, g_tar, od_topo_tar)

                print("Test with", config["tar_cities"][i], ".")
                with torch.no_grad():
                    if config["sample_method"] == "DDPM":
                        x_seq = model_F.p_sample_loop(x_0_tar.shape, condition_tar)
                    elif config["sample_method"] == "DDIM":
                        x_seq = model_F.DDIM_sample_loop(x_0_tar.shape, condition_tar)

                        if not os.path.exists("draft/check_ts/" + config["exp_name"]):
                            os.mkdir("draft/check_ts/" + config["exp_name"])
                        one_idx = 0
                        for one in x_seq:
                            rescale_od(one, data["tar"][i]["od_min_max"])
                            topo = condition_tar[3]
                            zero_flow_idx = topo == 0
                            one[zero_flow_idx] = -1
                            one = torch.clip(one, min=0)
                            one = one.round()
                            np.save("draft/check_ts/" + str(one_idx) + ".npy", one.numpy())

                    x_0_hat = rescale_od(x_seq[-1], data["tar"][i]["od_min_max"])
                    topo = condition_tar[3]
                    zero_flow_idx = topo == 0
                    x_0_hat[zero_flow_idx] = -1 # Zero flows are normalized to -1.
                    x_0_hat = torch.clip(x_0_hat, min=0)

                    x_0_hat, valid_x_0_tar = x_0_hat.round(), valid_x_0_tar.round()
                    logger.save_generation(config["tar_cities"][i], x_0_hat, "flow")
                    all_metrics = cal_all_metrics_flow(x_0_hat, valid_x_0_tar)
                    pprint(all_metrics)



def test():
    import json
    import dgl
    from model import FlowDiffusion, FlowNet
    from data_load import prepare_data
    config = json.load(open("exp/config/NYC_Chi.json", "r"))
    print(config)

    data = prepare_data(config)
    config["GAT_indim"] = data["src"]["attr"].shape[1]
    data["src"]["g"] = dgl.add_self_loop(data["src"]["g"])

    NN = FlowNet(config)
    model = FlowDiffusion(NN, config)
    train(config, model, data)




if __name__ == "__main__":
    test()