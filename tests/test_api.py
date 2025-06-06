import os, sys
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ['TESTING'] = '1'
from api import app

client = TestClient(app)


def test_courts_endpoint():
    response = client.get('/courts')
    assert response.status_code == 200
    data = response.json()
    assert 'total_courts' in data
    assert 'total_people' in data
    assert 'people_per_court' in data
    assert isinstance(data['people_per_court'], dict)


def test_courts_endpoint_use_camera_no_device():
    response = client.get('/courts', params={'use_camera': 'true'})
    assert response.status_code == 400
    data = response.json()
    assert data['detail'] == 'No camera detected'
