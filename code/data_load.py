import numpy as np
from sklearn.preprocessing import MinMaxScaler

from utils.tool import build_DGLGraph, scale_od, flow_to_topo_softlabel

def MinMaxer(data):
    return MinMaxScaler(feature_range=(-1, 1)).fit_transform(data)

def prepare_data(config):

    src_cities = config["src_cities"]
    tar_cities = config["tar_cities"]
    tar = {}

    # Multi source cities for training
    srcs = []
    for city in src_cities:
        src_path = "data/" + city + "/"
        src = {}
        src["attr"] = np.load(src_path + "attr.npy")
        src["dis"] = np.load(src_path + "dis.npy")
        src["g"] = build_DGLGraph(np.load(src_path + "adj.npy"))
        src["od"] = np.load(src_path + "od.npy")
        if config["topo_softlabel"] != 1:
            src["od_topo"] = np.zeros_like(src["od"])
            src["od_topo"][src["od"].nonzero()] = 1
        else:
            src["od_topo"] = flow_to_topo_softlabel(src["od"])
        srcs.append(src)

    # # source city
    # src_path = "data/" + config["src_city"] + "/"
    # src["attr"] = np.load(src_path + "attr.npy")
    # src["dis"] = np.load(src_path + "dis.npy")
    # src["g"] = build_DGLGraph(np.load(src_path + "adj.npy"))
    # src["od"] = np.load(src_path + "od.npy")
    # src["od_topo"] = np.zeros_like(src["od"])
    # src["od_topo"][src["od"].nonzero()] = 1

    # Multi target cities for test
    tars = []
    for city in tar_cities:
        tar_path = "data/" + city + "/"
        tar = {}
        tar["attr"] = np.load(tar_path + "attr.npy")
        tar["dis"] = np.load(tar_path + "dis.npy")
        tar["g"] = build_DGLGraph(np.load(tar_path + "adj.npy"))
        tar["od"] = np.load(tar_path + "od.npy")
        tar["od_topo"] = np.zeros_like(tar["od"])
        tar["od_topo"][tar["od"].nonzero()] = 1
        tars.append(tar)

    # # target city
    # tar_path = "data/" + config["tar_city"] + "/"
    # tar["attr"] = np.load(tar_path + "attr.npy")
    # tar["dis"] = np.load(tar_path + "dis.npy")
    # tar["g"] = build_DGLGraph(np.load(tar_path + "adj.npy"))
    # tar["od"] = np.load(tar_path + "od.npy")
    # tar["od_topo"] = np.zeros_like(tar["od"])
    # tar["od_topo"][tar["od"].nonzero()] = 1

    # normalization
    if config["attr_MinMax"] == 1:
        # src["attr"] = MinMaxer(src["attr"])
        for i in range(len(srcs)):
            srcs[i]["attr"] = MinMaxer(srcs[i]["attr"])
        # tar["attr"] = MinMaxer(tar["attr"])
        for i in range(len(tars)):
            tars[i]["attr"] = MinMaxer(tars[i]["attr"])
    if config["od_MinMax"] == 1:
        # src["od"], src["od_min_max"] = scale_od(src["od"])
        for i in range(len(srcs)):
            srcs[i]["od"], srcs[i]["od_min_max"] = scale_od(srcs[i]["od"])
        # tar["od"], tar["od_min_max"] = scale_od(tar["od"])
        for i in range(len(tars)):
            tars[i]["od"], tars[i]["od_min_max"] = scale_od(tars[i]["od"])

    data ={}
    # data["src"] = src
    data["src"] = srcs
    # data["tar"] = tar
    data["tar"] = tars
    
    return data
    

if __name__ == "__main__":
    test()