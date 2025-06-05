import requests
import unittest
import sys
import json
from datetime import datetime

class ThriveRemoteOSAPITester(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ThriveRemoteOSAPITester, self).__init__(*args, **kwargs)
        self.base_url = "https://6fdeb0af-cffd-41a8-8922-c43506e56eae.preview.emergentagent.com/api"
        self.tests_run = 0
        self.tests_passed = 0

    def setUp(self):
        self.tests_run += 1
        print(f"\n🔍 Running test: {self._testMethodName}")

    def tearDown(self):
        # Simplified tearDown
        if self._outcome.success:
            self.tests_passed += 1
            print(f"✅ Test passed")
        else:
            print(f"❌ Test failed")

    def test_root_endpoint(self):
        """Test the root API endpoint"""
        response = requests.get(f"{self.base_url}/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("timestamp", data)
        self.assertIn("features", data)
        print(f"API Version: {data.get('message')}")

    def test_system_performance(self):
        """Test the system performance API endpoint"""
        response = requests.get(f"{self.base_url}/system/performance")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success"))
        self.assertIn("performance", data)
        self.assertIn("timestamp", data)
        
        performance = data.get("performance", {})
        self.assertIn("cpu_usage", performance)
        self.assertIn("memory_usage", performance)
        self.assertIn("disk_usage", performance)
        print(f"System Performance - CPU: {performance.get('cpu_usage')}%, Memory: {performance.get('memory_usage')}%")

    def test_weather_enhanced(self):
        """Test the enhanced weather API endpoint"""
        locations = ["New York", "London", "Tokyo"]
        for location in locations:
            response = requests.get(f"{self.base_url}/weather/enhanced?location={location}")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data.get("success"))
            self.assertEqual(data.get("location"), location)
            self.assertIn("weather", data)
            
            weather = data.get("weather", {})
            self.assertIn("temperature", weather)
            self.assertIn("condition", weather)
            self.assertIn("humidity", weather)
            self.assertIn("forecast", weather)
            self.assertIn("air_quality", weather)
            
            # Check forecast data
            forecast = weather.get("forecast", [])
            self.assertEqual(len(forecast), 3)  # Should have 3-day forecast
            
            # Check air quality data
            air_quality = weather.get("air_quality", {})
            self.assertIn("aqi", air_quality)
            self.assertIn("category", air_quality)
            self.assertIn("pollutants", air_quality)
            
            print(f"Weather for {location}: {weather.get('temperature')}°C, {weather.get('condition')}")

    def test_news_live(self):
        """Test the live news API endpoint"""
        response = requests.get(f"{self.base_url}/news/live")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success"))
        self.assertIn("news", data)
        self.assertIn("total", data)
        
        news_items = data.get("news", [])
        self.assertGreater(len(news_items), 0)
        
        # Check first news item structure
        first_item = news_items[0]
        self.assertIn("title", first_item)
        self.assertIn("description", first_item)
        self.assertIn("source", first_item)
        self.assertIn("category", first_item)
        self.assertIn("published_at", first_item)
        
        print(f"Live News: {len(news_items)} items available")
        print(f"First news item: {first_item.get('title')}")

    def test_relocateme_opportunities(self):
        """Test the RelocateMe opportunities API endpoint"""
        response = requests.get(f"{self.base_url}/relocateme/opportunities")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success"))
        self.assertIn("opportunities", data)
        self.assertIn("total_opportunities", data)
        self.assertIn("featured_countries", data)
        
        opportunities = data.get("opportunities", [])
        self.assertGreater(len(opportunities), 0)
        
        # Check first opportunity structure
        first_opp = opportunities[0]
        self.assertIn("id", first_opp)
        self.assertIn("title", first_opp)
        self.assertIn("company", first_opp)
        self.assertIn("location", first_opp)
        self.assertIn("salary", first_opp)
        self.assertIn("relocation_package", first_opp)
        self.assertIn("benefits", first_opp)
        self.assertIn("requirements", first_opp)
        
        # Check relocation package
        relocation_package = first_opp.get("relocation_package", {})
        self.assertIn("visa_support", relocation_package)
        self.assertIn("moving_allowance", relocation_package)
        self.assertIn("temporary_housing", relocation_package)
        
        print(f"RelocateMe: {len(opportunities)} opportunities available")
        print(f"First opportunity: {first_opp.get('title')} at {first_opp.get('company')} in {first_opp.get('location')}")

    def test_downloads_api(self):
        """Test the downloads API endpoints"""
        # 1. Start a new download
        download_data = {
            "url": "https://example.com/sample.pdf",
            "filename": "test_document.pdf",
            "category": "documents"
        }
        
        response = requests.post(f"{self.base_url}/downloads/start", json=download_data)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success"))
        self.assertIn("download_id", data)
        
        download_id = data.get("download_id")
        print(f"Created download with ID: {download_id}")
        
        # 2. Get all downloads
        response = requests.get(f"{self.base_url}/downloads")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("downloads", data)
        downloads = data.get("downloads", [])
        self.assertGreater(len(downloads), 0)
        
        # 3. Update download progress
        progress_data = {
            "progress": 50.0,
            "status": "downloading"
        }
        response = requests.put(f"{self.base_url}/downloads/{download_id}/progress", json=progress_data)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success"))
        
        # 4. Get download status
        response = requests.get(f"{self.base_url}/downloads/{download_id}/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("download", data)
        download = data.get("download", {})
        self.assertEqual(download.get("id"), download_id)
        self.assertEqual(download.get("progress"), 50.0)
        self.assertEqual(download.get("status"), "downloading")
        
        # 5. Complete download
        progress_data = {
            "progress": 100.0,
            "status": "completed"
        }
        response = requests.put(f"{self.base_url}/downloads/{download_id}/progress", json=progress_data)
        self.assertEqual(response.status_code, 200)
        
        # 6. Verify completion
        response = requests.get(f"{self.base_url}/downloads/{download_id}/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        download = data.get("download", {})
        self.assertEqual(download.get("progress"), 100.0)
        self.assertEqual(download.get("status"), "completed")
        self.assertIn("completed_date", download)
        
        print(f"Download workflow tested successfully")

    def test_dashboard_stats(self):
        """Test the dashboard stats API endpoint"""
        response = requests.get(f"{self.base_url}/dashboard/stats")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check for required fields
        self.assertIn("productivity_score", data)
        self.assertIn("daily_streak", data)
        self.assertIn("savings_progress", data)
        self.assertIn("tasks_completed_today", data)
        
        print(f"Dashboard Stats - Productivity: {data.get('productivity_score')}, Streak: {data.get('daily_streak')} days")

    def test_dashboard_live_stats(self):
        """Test the live dashboard stats API endpoint"""
        response = requests.get(f"{self.base_url}/dashboard/live-stats")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check for required fields
        self.assertIn("remote_opportunities", data)
        self.assertIn("active_users", data)
        self.assertIn("uptime_hours", data)
        self.assertIn("database", data)
        
        print(f"Live Stats - Remote Opportunities: {data.get('remote_opportunities')}, Active Users: {data.get('active_users')}")

    def test_content_ai_tools(self):
        """Test the AI tools content API endpoint"""
        response = requests.get(f"{self.base_url}/content/ai-tools")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check for required fields
        self.assertIn("categories", data)
        self.assertIn("total_tools", data)
        self.assertIn("total_categories", data)
        self.assertIn("featured_tools", data)
        
        categories = data.get("categories", {})
        self.assertGreater(len(categories), 0)
        
        print(f"AI Tools - Total: {data.get('total_tools')}, Categories: {data.get('total_categories')}")

def run_tests():
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add tests to the suite
    suite.addTest(ThriveRemoteOSAPITester('test_root_endpoint'))
    suite.addTest(ThriveRemoteOSAPITester('test_system_performance'))
    suite.addTest(ThriveRemoteOSAPITester('test_weather_enhanced'))
    suite.addTest(ThriveRemoteOSAPITester('test_news_live'))
    suite.addTest(ThriveRemoteOSAPITester('test_relocateme_opportunities'))
    suite.addTest(ThriveRemoteOSAPITester('test_downloads_api'))
    suite.addTest(ThriveRemoteOSAPITester('test_dashboard_stats'))
    suite.addTest(ThriveRemoteOSAPITester('test_dashboard_live_stats'))
    suite.addTest(ThriveRemoteOSAPITester('test_content_ai_tools'))
    
    # Create a test runner
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Run the tests
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 Tests summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Tests passed: {result.testsRun - len(result.errors) - len(result.failures)}")
    print(f"Tests failed: {len(result.failures)}")
    print(f"Tests with errors: {len(result.errors)}")
    
    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(run_tests())
