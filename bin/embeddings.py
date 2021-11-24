import pandas as pd
import numpy as np
import os
from embeddings_reproduction import embedding_tools


def embed_sequences(data, model_path, out_path, k=5, overlap=False):
    embeds_data = []
    heavy = data["heavy"]
    light = data["light"]
    for seqlist in [heavy, light]:
        embeds = embedding_tools.get_embeddings(model_path, seqlist, k, overlap)
        embeds_data.append(pd.DataFrame.from_records(embeds))
    ncols = len(embeds_data[0].columns)
    embeds_pd = pd.concat(embeds_data, axis=1)
    embeds_pd.columns = [str(i) for i in range(ncols)] + [str(i) + "_2" for i in range(ncols)]
    embeds_pd.insert(0, "Ab_ID", data["Antibody_ID"])
    #embeds_pd.to_csv(out_path, index=False)
    embeds_pd.to_feather(out_path)
    return embeds_pd


def remove_excess(seq):
    seq = seq.replace("[", "").replace("]", "")
    seq = seq.replace("'", "")
    return seq.replace(",", "")


DATA_DIR = "../data"
MODEL_DIRECTORY = "../data/embed_models"
MODELS = [mod_file for mod_file in os.listdir(MODEL_DIRECTORY) if mod_file.endswith(".pkl")]


CHEN_DATA = pd.read_csv(os.path.join(DATA_DIR, "chen/deduplicated/chen_data.csv"))
TAP_DATA = pd.read_csv(os.path.join(DATA_DIR, "tap/TAP_data.csv"))


def embed_data(model, data, label):
    print(f"Creating embeddings with model {model}.")
    mod_name = model.split(".")[0]
    k = int(mod_name.split("_")[1])
    w = int(mod_name.split("_")[2])
    # 
    outfile = f"../data/{label}/embeddings/embeddings_{label}_{mod_name}.ftr"
    if not os.path.isfile(outfile):
        df = embed_sequences(data, os.path.join(MODEL_DIRECTORY, model), outfile, k=k)
        print(f"Writing embeddings to file {outfile}.")

        
if __name__ == "__main__":
    for model in MODELS: 
        embed_data(model, CHEN_DATA, "chen")
        embed_data(model, TAP_DATA, "tap")
