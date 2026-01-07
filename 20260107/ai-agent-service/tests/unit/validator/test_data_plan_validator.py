import unittest
from validators.data_plan_validator import validate_data_plan

class TestDataPlanValidator(unittest.TestCase):
    """Unit tests for the data plan validator."""

    def setUp(self):
        """Set up a valid data plan structure for tests."""
        self.valid_plan = {
            "service_name": "Test Service",
            "data_driven_features": [{
                "idea_name": "Feature A", "description": "Desc A", "required_data": ["data1"]
            }],
            "inferred_insights": [{
                "idea_name": "Insight B", "description": "Desc B", "required_sensors": ["sensor1"]
            }],
            "new_data_sources": [{
                "source_type": "Type C", "source_name": "Name C",
                "collectable_data": "Data C", "value_proposition": "Value C"
            }]
        }

    def test_validate_data_plan_with_valid_data(self):
        """Tests that a valid data plan passes validation."""
        print("\nRunning test: test_validate_data_plan_with_valid_data")
        result = validate_data_plan(self.valid_plan)
        self.assertIsNotNone(result)
        self.assertEqual(result['service_name'], "Test Service")

    def test_validate_data_plan_with_invalid_data(self):
        """Tests that an invalid data plan fails validation and returns None."""
        print("\nRunning test: test_validate_data_plan_with_invalid_data")
        invalid_plan = self.valid_plan.copy()
        # Make a nested part invalid by removing a required field
        del invalid_plan["inferred_insights"][0]["required_sensors"]

        result = validate_data_plan(invalid_plan)
        self.assertIsNone(result)

    def test_validate_data_plan_with_non_dict_input(self):
        """Tests that a non-dictionary input returns None."""
        print("\nRunning test: test_validate_data_plan_with_non_dict_input")
        result = validate_data_plan([1, 2, 3])  # Using a list instead of a dict
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()