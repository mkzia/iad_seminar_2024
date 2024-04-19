from typing import Union
import dill
from fastapi import FastAPI

app = FastAPI()

with open('rfr_v1.pkl', 'rb') as f: 
    model = dill.load(f) 


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}