from fastapi.testclient import TestClient
import pytest

def test_predict_post(app_client: TestClient):
    """
    Makes sure the API and the POST method for making predictions
    are working correctly.
    """
    
    input_value = {"values": [10.0]}
    ground_truth_data = [[30.99516]]

    response = app_client.post(
        "/",
        headers={"Content-Type":"application/json"},
        json=input_value
    )
    
    assert response.status_code == 200
    response_data = response.json()["predictions"]
    for respose_item, gt_item in zip(response_data, ground_truth_data):
        assert pytest.approx(respose_item, 0.001) == gt_item
            
def test_predict_post_batch(app_client: TestClient):
    """
    Makes sure the API and the POST method for making predictions
    in batches are working correctly.
    """
    
    input_value = {"values": [10.0, 5.0]}
    ground_truth_data = [[30.99516], [15.998667]]
    
    response = app_client.post(
        "/",
        headers={"Content-Type":"application/json"},
        json=input_value
    )
    
    assert response.status_code == 200
    response_data = response.json()["predictions"]
    for respose_item, gt_item in zip(response_data, ground_truth_data):
        assert pytest.approx(respose_item, 0.001) == gt_item