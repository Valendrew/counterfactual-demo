from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import torch

from counterfactual.utils import util_counterfactual, util_models


# create the app
app = FastAPI()

# add the CORS middleware
origins = ["http://127.0.0.1:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# create the data model
class Data(BaseModel):
    name: str

class Counterfactual(BaseModel):
    id: int

# read the data
df = pd.read_csv("counterfactual/data/processed/gsm_full.csv", sep=",")
df_model = pd.read_csv("counterfactual/data/model/best_train.csv", sep=",")
X, y = df_model.drop("misc_price", axis=1), df_model["misc_price"]

# # load the model
num_class = len(np.unique(y))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nn_model = util_models.NNClassification([128, 128], X.shape[1], num_class=num_class).to(
    device
)
nn_model.load_state_dict(torch.load("counterfactual/models/nn_model.pt")["model_state_dict"])

# # create the counterfactual model
continuous_feat = [
    "display_size",
    "battery",
    "memory_ram_gb",
    "memory_rom_gb",
    "main_camera_resolution",
    "selfie_camera_resolution",
]
omlt_model = util_counterfactual.OmltCounterfactual(
    X, y, nn_model, continuous_feat=continuous_feat
)


# create the root endpoint
@app.get("/")
async def root():
    return {"message": "Hello World"}


# create the search endpoint
@app.post("/search")
async def search(data: Data):
    # filter the data
    filtered = df[df["oem_model"].str.contains(data.name, case=False)]
    return filtered.to_json(orient="index")


# @app.post("/counterfactual")
# async def counterfactual(counterfactual: Counterfactual):
#     # take the sample from the data
#     idx = counterfactual.id

#     sample_feat = X.iloc[idx]
#     sample_label = y.iloc[idx]

#     # check the prediction
#     pred, logits, softmax = util_models.evaluate_sample(nn_model, sample_feat, device)
#     if pred != sample_label:
#         return {"message": "The prediction is wrong"}

#     # get the counterfactual
#     cf_class = util_counterfactual.get_counterfactual_class(sample_label, num_class, lower=True)
#     min_prob = 0.51
#     obj_weights = [1, 0.5, 0.8]
#     solver_options = {"timelimit": 120}
#     cf = omlt_model.generate_counterfactuals(
#         sample_feat.values,
#         cf_class,
#         min_prob,
#         obj_weights,
#         solver_options=solver_options,
#         verbose=True,
#     )

#     # check if the counterfactual is valid
#     cf_sample = cf.loc[0, :]
#     cf_pred, cf_logits, cf_softmax = util_models.evaluate_sample(
#         nn_model, cf_sample, device
#     )
#     if cf_pred != cf_class:
#         return {"message": "The counterfactual is not valid"}

#     # return the counterfactual
#     return cf_sample.to_list()
