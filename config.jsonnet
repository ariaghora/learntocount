local n_classes = 3;
local n_views = 3;
local batch_size = 20;
local image_size = 256;
local object_size = 64;
local max_epochs = 100;


local generate_train_data_def = {
    "n_classes": n_classes,
    "n_per_class": 20,
    "n_views": n_views,
    "image_size": image_size,
    "object_size": object_size,
};

local generate_test_data_def = {
    "n_classes": n_classes,
    "n_per_class": 5,
    "n_views": n_views,
    "image_size": image_size,
    "object_size": object_size,
};


local load_train_data_def = {
    "n_views": n_views,
    "batch_size": batch_size,
};

local load_test_data_def = {
    "n_views": n_views,
    "batch_size": batch_size,
};

local train_model_def = {
    "train_data_loader": "ref::load_train_data",
    "max_epochs": max_epochs,
    "n_views": n_views,
    "n_classes": n_classes
};

{
    "node_implementation_modules": [
        "steps.data_generation",
        "steps.data_engineering",
        "steps.data_science"
    ],

    "data_catalog": import "data_catalog.libsonnet",

    "pipeline_definition": {
        "_default" : {},

        "data_generation_pipeline": {
            "generate_train_data": generate_train_data_def,
            "generate_test_data": generate_test_data_def
        },

        "training_pipeline": {
            "load_train_data": load_train_data_def,
            "load_test_data": load_test_data_def,
            "train_model": train_model_def,
            "evaluate_model": {
                "test_data_loader": "ref::load_test_data",
                "model": "ref::train_model"
            },
        },

        "evaluation_pipeline": {
            "load_test_data": load_test_data_def,
            "evaluate_model": {
                "test_data_loader": "ref::load_test_data",
                "model": "data::trained_model"
            },
        },
    },
}