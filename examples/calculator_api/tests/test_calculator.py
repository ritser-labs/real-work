import pytest
import requests
import json
import time
from threading import Thread
import subprocess
import os
import signal


class TestCalculatorAPI:
    """Test suite for the calculator API"""
    
    @classmethod
    def setup_class(cls):
        """Set up test environment"""
        cls.base_url = "http://localhost:5000"
        cls.app_process = None
        
        # Try to start the Flask app if it's not running
        try:
            requests.get(f"{cls.base_url}/health", timeout=2)
            print("App is already running")
        except requests.exceptions.RequestException:
            print("Starting Flask app...")
            # Start the Flask app in background
            cls.app_process = subprocess.Popen(
                ["python", "app.py"],
                cwd="/workspace",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            time.sleep(3)  # Wait for app to start
    
    @classmethod
    def teardown_class(cls):
        """Clean up test environment"""
        if cls.app_process:
            cls.app_process.terminate()
            cls.app_process.wait()
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = requests.get(f"{self.base_url}/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}
    
    def test_addition(self):
        """Test addition operation"""
        data = {
            "operation": "add",
            "numbers": [1, 2, 3, 4]
        }
        response = requests.post(f"{self.base_url}/calculate", json=data)
        assert response.status_code == 200
        assert response.json() == {"result": 10}
    
    def test_subtraction(self):
        """Test subtraction operation"""
        data = {
            "operation": "subtract",
            "numbers": [10, 3, 2]
        }
        response = requests.post(f"{self.base_url}/calculate", json=data)
        assert response.status_code == 200
        assert response.json() == {"result": 5}
    
    def test_multiplication(self):
        """Test multiplication operation"""
        data = {
            "operation": "multiply",
            "numbers": [2, 3, 4]
        }
        response = requests.post(f"{self.base_url}/calculate", json=data)
        assert response.status_code == 200
        assert response.json() == {"result": 24}
    
    def test_division(self):
        """Test division operation"""
        data = {
            "operation": "divide",
            "numbers": [20, 2, 2]
        }
        response = requests.post(f"{self.base_url}/calculate", json=data)
        assert response.status_code == 200
        assert response.json() == {"result": 5.0}
    
    def test_division_by_zero(self):
        """Test division by zero error handling"""
        data = {
            "operation": "divide",
            "numbers": [10, 0]
        }
        response = requests.post(f"{self.base_url}/calculate", json=data)
        assert response.status_code == 400
        assert "error" in response.json()
        assert "division by zero" in response.json()["error"].lower()
    
    def test_invalid_operation(self):
        """Test invalid operation error handling"""
        data = {
            "operation": "invalid_op",
            "numbers": [1, 2, 3]
        }
        response = requests.post(f"{self.base_url}/calculate", json=data)
        assert response.status_code == 400
        assert "error" in response.json()
    
    def test_empty_numbers(self):
        """Test empty numbers array"""
        data = {
            "operation": "add",
            "numbers": []
        }
        response = requests.post(f"{self.base_url}/calculate", json=data)
        assert response.status_code == 400
        assert "error" in response.json()
    
    def test_single_number_operations(self):
        """Test operations with single number"""
        # Addition with single number
        data = {
            "operation": "add",
            "numbers": [5]
        }
        response = requests.post(f"{self.base_url}/calculate", json=data)
        assert response.status_code == 200
        assert response.json() == {"result": 5}
        
        # Multiplication with single number
        data = {
            "operation": "multiply",
            "numbers": [7]
        }
        response = requests.post(f"{self.base_url}/calculate", json=data)
        assert response.status_code == 200
        assert response.json() == {"result": 7}
    
    def test_invalid_json(self):
        """Test invalid JSON payload"""
        response = requests.post(
            f"{self.base_url}/calculate",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 400
        assert "error" in response.json()
    
    def test_missing_fields(self):
        """Test missing required fields"""
        # Missing operation
        data = {"numbers": [1, 2, 3]}
        response = requests.post(f"{self.base_url}/calculate", json=data)
        assert response.status_code == 400
        assert "error" in response.json()
        
        # Missing numbers
        data = {"operation": "add"}
        response = requests.post(f"{self.base_url}/calculate", json=data)
        assert response.status_code == 400
        assert "error" in response.json()
    
    def test_non_numeric_values(self):
        """Test non-numeric values in numbers array"""
        data = {
            "operation": "add",
            "numbers": [1, "not_a_number", 3]
        }
        response = requests.post(f"{self.base_url}/calculate", json=data)
        assert response.status_code == 400
        assert "error" in response.json()
    
    def test_large_numbers(self):
        """Test with large numbers"""
        data = {
            "operation": "add",
            "numbers": [1000000, 2000000, 3000000]
        }
        response = requests.post(f"{self.base_url}/calculate", json=data)
        assert response.status_code == 200
        assert response.json() == {"result": 6000000}
    
    def test_decimal_numbers(self):
        """Test with decimal numbers"""
        data = {
            "operation": "add",
            "numbers": [1.5, 2.5, 3.0]
        }
        response = requests.post(f"{self.base_url}/calculate", json=data)
        assert response.status_code == 200
        assert response.json() == {"result": 7.0}
    
    def test_negative_numbers(self):
        """Test with negative numbers"""
        data = {
            "operation": "add",
            "numbers": [-1, -2, 3]
        }
        response = requests.post(f"{self.base_url}/calculate", json=data)
        assert response.status_code == 200
        assert response.json() == {"result": 0} 