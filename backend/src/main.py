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


TEST_RAW_PATH = "counterfactual/data/processed/gsm_test.csv"
TEST_MODEL_PATH = "counterfactual/data/model/gsm_test.csv"
TRAIN_MODEL_PATH = "counterfactual/data/model/gsm_train.csv"
NN_WEIGHTS_PATH = "counterfactual/models/nn_model.pt"

# Read both the standardised and raw data, removing from the latter the rows that are not in the former
MODEL_DATA = pd.read_csv(TEST_MODEL_PATH, sep=",", index_col=0)
RAW_DATA = pd.read_csv(TEST_RAW_PATH, sep=",", index_col=0).loc[MODEL_DATA.index, :]

# Read the train data
TRAIN_MODEL_DATA = pd.read_csv(TRAIN_MODEL_PATH, sep=",", index_col=0)
TRAIN_FEAT_DATA = TRAIN_MODEL_DATA.drop("misc_price", axis=1)
TRAIN_LABEL_DATA = TRAIN_MODEL_DATA.misc_price

# Load the pytorch model
DROPOUT_RATE = 0.2
HIDDEN_LAYERS = [64, 64]
NUM_CLASSES = MODEL_DATA.misc_price.unique().shape[0]
NUM_FEATURES = MODEL_DATA.shape[1] - 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NN_MODEL = util_models.NNClassification(
    hidden_dims=HIDDEN_LAYERS,
    num_feat=NUM_FEATURES,
    num_class=NUM_CLASSES,
    dropout_rate=DROPOUT_RATE,
).to(device)
# Load the saved weights into the model
NN_MODEL.load_state_dict(torch.load(NN_WEIGHTS_PATH)["model_state_dict"])

# Load scikit-learn pipeline
FEATURES_PIPELINE: ColumnTransformer = pickle.load(
    open("counterfactual/config/pipeline.pkl", "rb")
)

# OMLT parameters dictionary for the counterfactual
DICE_DICT = {
    "target" : "misc_price",
    "n_cf" : 1,
    "verbose": False
}


CONTINUOUS_FEAT = [
    "display_size", "battery", "memory_ram_gb", 
    "memory_rom_gb", "main_camera_resolution", "selfie_camera_resolution"]
CATEGORICAL_FEAT = TRAIN_FEAT_DATA.columns.drop(CONTINUOUS_FEAT).tolist()

CONTINUOUS_FEAT = [(TRAIN_FEAT_DATA.columns.get_loc(feat), feat) for feat in CONTINUOUS_FEAT]
CATEGORICAL_FEAT = [(TRAIN_FEAT_DATA.columns.get_loc(feat), feat) for feat in CATEGORICAL_FEAT]

WEIGHTS = np.repeat(1, len(CONTINUOUS_FEAT) + len(CATEGORICAL_FEAT))

FEATURE_PROPS = util_counterfactual.create_feature_props(TRAIN_FEAT_DATA, CONTINUOUS_FEAT, CATEGORICAL_FEAT, WEIGHTS)


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
        NN_MODEL, X_sample, y_sample, verbose=False, device=device
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
    if not isinstance(sample, pd.DataFrame):
        return {"message": "The sample is not a DataFrame object"}

    X_sample, y_sample = (
        sample.drop("misc_price", axis=1),
        sample.misc_price.astype(int)
    )

    # Get the counterfactual
    try:
        cfs = util_counterfactual.generate_counterfactuals_from_sample_list(
            NN_MODEL, "dice", TRAIN_FEAT_DATA, TRAIN_LABEL_DATA, X_sample, y_sample,
            feature_props=FEATURE_PROPS, type_cf=data.target, backend="PYT", target_column="misc_price",
            dice_method="random", pipeline=FEATURES_PIPELINE, save_filename=None, **DICE_DICT,
        )
    except Exception as e:
        print(e)
        return {"message": "The counterfactual could not be generated"}
    
    assert isinstance(cfs, pd.DataFrame), "The counterfactual is not a DataFrame object"
    
    cf_only = cfs.iloc[0]
    price_min, price_max = cf_only.pop("misc_price_min"), cf_only.pop("misc_price_max")
    cf_only["misc_price"] = f"{price_min} - {price_max}"
    return json.loads(cf_only.to_json())