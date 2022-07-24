{
    "generator_obj_book": {
        "path": "data/01_raw/generator/generator_obj_book.png",
        "type": "DT_PIL_IMAGE",
    },
    "generator_obj_chair": {
        "path": "data/01_raw/generator/generator_obj_chair.png",
        "type": "DT_PIL_IMAGE",
    },
    "generator_obj_bottle": {
        "path": "data/01_raw/generator/generator_obj_bottle.png",
        "type": "DT_PIL_IMAGE",
    },
    "generated_dataset_metadata": {
        "path": "data/02_intermediate/generated_dataset_metadata.csv",
        "type": "DT_PANDAS_DATAFRAME",
        "write_params": {
            "index": false,
            "header": true,
        },
    },

    "trained_model": {
        "path": "data/03_output/model.pt",
        "type": "DT_PYTORCH_MODEL",
        "read_params": {
            "n_views": 3,
            "n_classes": 3
        },
    },

}