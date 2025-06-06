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


def test_court_count_endpoint():
    response = client.get('/court_count')
    assert response.status_code == 200
    data = response.json()
    assert 'total_courts' in data
    assert isinstance(data['total_courts'], int)


def test_court_count_use_camera_no_device():
    response = client.get('/court_count', params={'use_camera': 'true'})
    assert response.status_code == 400


def test_status_endpoint():
    response = client.get('/status')
    assert response.status_code == 200
    data = response.json()
    assert 'is_raspberry_pi' in data
    assert 'camera_available' in data

