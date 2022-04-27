from typing import List

from pydantic import BaseModel


class InputData(BaseModel):

    """
    Represents the expected body to the `predict` endpoint.
    """

    data: List[int]

    # Add example schema for OpenAPI
    class Config:
        schema_extra = {"data": [10]}


class ResponseData(BaseModel):

    """
    Represents the response to the `predict` endpoint
    """

    prediction: float

    # Add example schema for OpenAPI
    class Config:
        schema_extra = {"prediction": 30.102}
