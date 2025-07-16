import pytest
import json
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """Test health check endpoint"""
    rv = client.get('/health')
    assert rv.status_code == 200
    assert rv.get_json() == {'status': 'ok'}

def test_add(client):
    """Test addition endpoint"""
    rv = client.post('/calculate', json={'operation': 'add', 'numbers': [1, 2, 3]})
    assert rv.status_code == 200
    assert rv.get_json() == {'result': 6}

def test_subtract(client):
    """Test subtraction endpoint"""
    rv = client.post('/calculate', json={'operation': 'subtract', 'numbers': [10, 2, 3]})
    assert rv.status_code == 200
    assert rv.get_json() == {'result': 5}

def test_multiply(client):
    """Test multiplication endpoint"""
    rv = client.post('/calculate', json={'operation': 'multiply', 'numbers': [2, 3, 4]})
    assert rv.status_code == 200
    assert rv.get_json() == {'result': 24}

def test_divide(client):
    """Test division endpoint"""
    rv = client.post('/calculate', json={'operation': 'divide', 'numbers': [20, 2, 5]})
    assert rv.status_code == 200
    assert rv.get_json() == {'result': 2.0}

def test_divide_by_zero(client):
    """Test division by zero"""
    rv = client.post('/calculate', json={'operation': 'divide', 'numbers': [10, 0]})
    assert rv.status_code == 400
    assert 'error' in rv.get_json()

def test_invalid_operation(client):
    """Test invalid operation"""
    rv = client.post('/calculate', json={'operation': 'invalid', 'numbers': [1, 2]})
    assert rv.status_code == 400
    assert 'error' in rv.get_json() 