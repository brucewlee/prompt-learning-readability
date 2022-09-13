import pandas as pd
import ndjson

supported_file_identifiers = [
    "tsv",
    "csv",
    "json"
]

#cannot differentiate ndjson and json for now
def detect_file_identifier(path: str) -> str:
    for file_identifier in supported_file_identifiers:
        identifier_length = len(file_identifier)
        if path[-identifier_length:] == file_identifier:
            return file_identifier
    return "not supported"

def load_ndjson_and_return_pandas(load_path: str) -> pd.DataFrame:
    with open(load_path) as f:
        ndjson_data = ndjson.load(f)
    df = pd.DataFrame(ndjson_data)
    return df

def load_json_and_return_pandas(load_path: str) -> pd.DataFrame:
    df = pd.read_json(load_path)
    return df
    
def get_pandas_and_save_ndjson(df: pd.DataFrame, save_path: str) -> None:   
    df.to_json(f'{save_path}', orient="records", lines=True)

def load_csv_and_return_pandas(load_path: str) -> pd.DataFrame:
    df = pd.read_csv(load_path)
    return df