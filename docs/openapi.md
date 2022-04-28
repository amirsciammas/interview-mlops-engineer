# MLOps Engineer Predictions API

API to make predictions using a Tensorflow model! ðŸš€

## Predict

You can perform predictions!

## Version: 0.0.1

**License:** MIT License

### /status

#### GET
##### Summary

Status

##### Description

Healthcheck for the API

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Successful Response |

### /predict

#### POST
##### Summary

Predict Model

##### Description

Performs the prediction with the loaded model usingthe request data

##### Responses

| Code | Description |
| ---- | ----------- |
| 200 | Successful Response |
| 422 | Validation Error |

### Models

#### HTTPValidationError

| Name | Type | Description | Required |
| ---- | ---- | ----------- | -------- |
| detail | [ object ] |  | No |

#### InputData

Represents the expected body to the `predict` endpoint.

| Name | Type | Description | Required |
| ---- | ---- | ----------- | -------- |
| data | [ integer ] |  | Yes |

#### ResponseData

Represents the response to the `predict` endpoint

| Name | Type | Description | Required |
| ---- | ---- | ----------- | -------- |
| prediction | [ number ] |  | Yes |

#### ValidationError

| Name | Type | Description | Required |
| ---- | ---- | ----------- | -------- |
| loc | [  ] |  | Yes |
| msg | string |  | Yes |
| type | string |  | Yes |
