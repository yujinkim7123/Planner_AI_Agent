import unittest
from validators.persona_validator import validate_personas

class TestPersonaValidator(unittest.TestCase):

    def setUp(self):
        self.valid_persona_1 = {
            "name": "Kim Minjun",
            "role": "Tech-savvy professional",
            "demographics": "Early 30s, single",
            "behavioral_traits": ["Loves new gadgets"],
            "needs_and_goals": ["Stay productive"],
            "pain_points": ["Complex user interfaces"]
        }
        self.valid_persona_2 = {
            "name": "Lee Hana",
            "role": "Busy parent",
            "demographics": "Late 30s, married with kids",
            "behavioral_traits": ["Values convenience"],
            "needs_and_goals": ["Manage family schedule"],
            "pain_points": ["Not enough time"]
        }
        self.invalid_persona = {
            # Missing the required 'name' field
            "role": "Incomplete persona",
            "demographics": "N/A"
        }

    def test_validate_personas_with_valid_data(self):
        
        print("\nRunning test: test_validate_personas_with_valid_data")
        personas_data = [self.valid_persona_1, self.valid_persona_2]
        result = validate_personas(personas_data)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['name'], self.valid_persona_1['name'])

    def test_validate_personas_with_invalid_data(self):
       
        print("\nRunning test: test_validate_personas_with_invalid_data")
        personas_data = [self.valid_persona_1, self.invalid_persona]
        result = validate_personas(personas_data)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['name'], self.valid_persona_1['name'])
        
    def test_validate_personas_with_non_list_input(self):

        print("\nRunning test: test_validate_personas_with_non_list_input")
        result = validate_personas({"not": "a list"})
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

if __name__ == '__main__':
    unittest.main()