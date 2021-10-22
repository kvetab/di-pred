import pandas as pd
import numpy as np
import os
from embeddings_reproduction import embedding_tools


def embed_sequences(data, model_path, out_path, k=5, overlap=False):
    embeds_data = []
    split = data["Antibody"].str.split()
    heavy = [remove_excess(seqs[0]) for seqs in split]
    light = [remove_excess(seqs[1]) for seqs in split]
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


MODEL_DIRECTORY = "../embed_models"
MODELS = [mod_file for mod_file in os.listdir(MODEL_DIRECTORY) if mod_file.endswith(".pkl")]

DATA_TRAIN = pd.read_csv("data/chen_train_data.csv", sep=";")
DATA_VALID = pd.read_csv("data/chen_valid_data.csv", sep=";")
DATA_TEST = pd.read_csv("data/chen_test_data.csv", sep=";")


def embed_data(model, data_train, data_test, data_valid):
    print(f"Creating embeddings with model {model}.")
    mod_name = model.split(".")[0]
    k = int(mod_name.split("_")[1])
    w = int(mod_name.split("_")[2])
    # train
    outfile = f"data/embeddings/embeddings_train_{mod_name}.ftr"
    if not os.path.isfile(outfile):
        df = embed_sequences(data_train, f"../embed_models/{model}", outfile, k=k)
        print(f"Writing embeddings to file {outfile}.")
    # validation
    outfile = f"data/embeddings/embeddings_valid_{mod_name}.ftr"
    if not os.path.isfile(outfile):
        df = embed_sequences(data_valid, f"../embed_models/{model}", outfile, k=k)
        print(f"Writing embeddings to file {outfile}.")
    # test
    outfile = f"data/embeddings/embeddings_test_{mod_name}.ftr"
    if not os.path.isfile(outfile):
        df = embed_sequences(data_test, f"../embed_models/{model}", outfile, k=k)
        print(f"Writing embeddings to file {outfile}.")

        
if __name__ == "__main__":
    for model in MODELS: 
        embed_data(model, DATA_TRAIN, DATA_TEST, DATA_VALID)
