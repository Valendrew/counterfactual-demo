import pickle

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
import torch

import sys
sys.path.insert(0, "counterfactual")
from utils import util_counterfactual, util_models


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
class SearchData(BaseModel):
    name: str


class ModelData(BaseModel):
    model_id: int


RAW_PATH = "counterfactual/data/processed/gsm_test.csv"
MODEL_PATH = "counterfactual/data/model/gsm_test.csv"
MODEL_WEIGHTS_PATH = "counterfactual/models/nn_model.pt"

# Read both the standardised and raw data, removing from the latter the rows that are not in the former
model_data = pd.read_csv(MODEL_PATH, sep=",", index_col=0)
raw_data = pd.read_csv(RAW_PATH, sep=",", index_col=0).loc[model_data.index, :]

# df_model = pd.read_csv("counterfactual/data/model/best_train.csv", sep=",")
# X, y = df_model.drop("misc_price", axis=1), df_model["misc_price"]

# Load the pytorch model
hidden_layers = [128, 128, 128]
num_of_classes = model_data.misc_price.unique().shape[0]
num_of_features = model_data.shape[1] - 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nn_model = util_models.NNClassification(
    hidden_dims=hidden_layers,
    num_feat=num_of_features,
    num_class=num_of_classes,
    dropout_rate=0.3,
).to(device)
# Load the saved weights into the model
nn_model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH)["model_state_dict"])

# Load scikit-learn pipeline
features_pipeline: ColumnTransformer = pickle.load(open("counterfactual/config/pipeline.pkl", "rb"))

# Create the counterfactual model
# continuous_feat = [
#     "display_size",
#     "battery",
#     "memory_ram_gb",
#     "memory_rom_gb",
#     "main_camera_resolution",
#     "selfie_camera_resolution",
# ]
# omlt_model = util_counterfactual.OmltCounterfactual(
#     X, y, nn_model, continuous_feat=continuous_feat
# )


# create the root endpoint
@app.get("/")
async def root():
    return {"message": "Hello World"}


# create the search endpoint
@app.post("/search")
async def search(data: SearchData):
    # filter the data
    boolean_mask = raw_data.oem_model.str.contains(data.name, case=False)
    filtered = raw_data[boolean_mask].sort_values(by=["oem_model"], ascending=False)
    return filtered.to_json(orient="index")


@app.post("/inference")
async def inference(data: ModelData):
    # get the sample
    sample = model_data.loc[data.model_id, :]
    X_sample, y_sample = sample.drop("misc_price", axis=0), sample["misc_price"]
    # get the prediction
    y_prediction = util_models.evaluate_sample(nn_model, X_sample, y_sample, device)
    y_prediction_df = pd.DataFrame({"misc_price": y_prediction}, index=[data.model_id])
    y_inverse_prediction = features_pipeline.transformers_[0][1].inverse_transform(y_prediction_df)
    y_inverse_prediction: pd.Series = y_inverse_prediction.loc[data.model_id, :]

    min_price = y_inverse_prediction[0]
    max_price = y_inverse_prediction[1]
    # return the prediction
    return {"prediction": y_prediction, "min_price": min_price, "max_price": max_price}


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
