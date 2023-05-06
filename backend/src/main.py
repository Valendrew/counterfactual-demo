import json
import pickle

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pandas.io.formats.style
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import torch

import sys

sys.path.insert(0, "counterfactual")
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
class SearchData(BaseModel):
    name: str


class ModelData(BaseModel):
    model_id: int


class CounterFactualData(BaseModel):
    model_id: int
    target: str


RAW_PATH = "counterfactual/data/processed/gsm_test.csv"
MODEL_PATH = "counterfactual/data/model/gsm_test.csv"
MODEL_WEIGHTS_PATH = "counterfactual/models/nn_model.pt"

TRAIN_MODEL_PATH = "counterfactual/data/model/gsm_train.csv"

# Read both the standardised and raw data, removing from the latter the rows that are not in the former
MODEL_DATA = pd.read_csv(MODEL_PATH, sep=",", index_col=0)
RAW_DATA = pd.read_csv(RAW_PATH, sep=",", index_col=0).loc[MODEL_DATA.index, :]

# Read the train data
TRAIN_MODEL_DATA = pd.read_csv(TRAIN_MODEL_PATH, sep=",", index_col=0)

# df_model = pd.read_csv("counterfactual/data/model/best_train.csv", sep=",")
# X, y = df_model.drop("misc_price", axis=1), df_model["misc_price"]

# Load the pytorch model
hidden_layers = [128, 128, 128]
num_of_classes = MODEL_DATA.misc_price.unique().shape[0]
num_of_features = MODEL_DATA.shape[1] - 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NN_MODEL = util_models.NNClassification(
    hidden_dims=hidden_layers,
    num_feat=num_of_features,
    num_class=num_of_classes,
    dropout_rate=0.3,
).to(device)
# Load the saved weights into the model
NN_MODEL.load_state_dict(torch.load(MODEL_WEIGHTS_PATH)["model_state_dict"])

# Load scikit-learn pipeline
FEATURES_PIPELINE: ColumnTransformer = pickle.load(
    open("counterfactual/config/pipeline.pkl", "rb")
)

# OMLT parameters dictionary for the counterfactual
OMLT_DICT = {
    "min_probability": 0.6,
    "obj_weights": [1, 0.8, 0.7],
    "cont_weights": [1, 1, 1, 1, 1, 1],
    "cat_weights": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "solver": "multistart",
    # "solver": "mindtpy",
    "solver_options": {"timelimit": 100, "strategy": "rand_guess_and_bound"},
    "verbose": True,
}

CONTINUOUS_FEATURES = [
    "display_size",
    "battery",
    "memory_ram_gb",
    "memory_rom_gb",
    "main_camera_resolution",
    "selfie_camera_resolution",
]


# create the root endpoint
@app.get("/")
async def root():
    return {"message": "Hello World"}


# create the search endpoint
@app.post("/search")
async def search(data: SearchData):
    # Filter the data
    boolean_mask = RAW_DATA.oem_model.str.contains(data.name, case=False)
    filtered = RAW_DATA[boolean_mask].sort_values(by=["oem_model"], ascending=False)
    return json.loads(filtered.to_json(orient="index"))


@app.post("/inference")
async def inference(data: ModelData):
    # Get the sample
    sample = MODEL_DATA.loc[data.model_id, :]
    assert isinstance(sample, pd.Series), "The sample is not a Series object"
    X_sample, y_sample = sample.drop("misc_price", axis=0), int(sample.misc_price)
    # get the prediction
    y_prediction = util_models.evaluate_sample(
        NN_MODEL, X_sample, y_sample, device, verbose=False
    )

    # Get the label pipeline
    label_pipeline = FEATURES_PIPELINE.named_transformers_["label"]
    assert isinstance(
        label_pipeline, Pipeline
    ), "The label pipeline is not a Pipeline object"

    # Create a dataframe with the prediction for the inverse transform
    y_prediction_df = pd.DataFrame({"misc_price": y_prediction}, index=[data.model_id])
    y_inverse_prediction = label_pipeline.inverse_transform(y_prediction_df)
    y_inverse_prediction = y_inverse_prediction.loc[data.model_id, :]

    # The inverse transform returns a dataframe with two columns: min_price and max_price
    min_price = y_inverse_prediction[0]
    max_price = y_inverse_prediction[1]

    return {
        "prediction": y_prediction,
        "ground_truth": y_sample,
        "min_price": min_price,
        "max_price": max_price,
    }


@app.post("/counterfactual")
async def counterfactual(data: CounterFactualData):
    # Get the sample
    sample = MODEL_DATA.loc[[data.model_id], :]
    assert isinstance(sample, pd.DataFrame), "The sample is not a DataFrame object"
    X_sample, y_sample = (
        sample.drop("misc_price", axis=1),
        sample.misc_price.astype(int).item(),
    )

    # Get the counterfactual
    try:
        cfs = util_counterfactual.generate_counterfactual_from_sample(
            NN_MODEL,
            cf_type="omlt",
            X_train=TRAIN_MODEL_DATA.drop("misc_price", axis=1),
            y_train=TRAIN_MODEL_DATA.misc_price,
            sample=X_sample,
            sample_label=y_sample,
            cont_feat=CONTINUOUS_FEATURES,
            type_cf=data.target,
            pipeline=FEATURES_PIPELINE,
            **OMLT_DICT,
        )
    except Exception as e:
        print(e)
        return {"message": "The counterfactual could not be generated"}
    
    assert isinstance(cfs, list), "The counterfactual is not a list"
    assert len(cfs) > 0, "The counterfactual is empty"
    if isinstance(cfs[0], pandas.io.formats.style.Styler):
        cf = cfs[0].data
    elif isinstance(cfs[0], pd.DataFrame):
        cf = cfs[0]
    else:
        raise TypeError("The counterfactual is not a DataFrame object")

    assert isinstance(cf, pd.DataFrame), "The counterfactual is not a DataFrame object"
    
    cf_only = cf.T.loc["Counterfactual_0"]
    price_min, price_max = cf_only.pop("misc_price_min"), cf_only.pop("misc_price_max")
    cf_only["misc_price"] = f"{price_min} - {price_max}"
    return json.loads(cf_only.to_json())