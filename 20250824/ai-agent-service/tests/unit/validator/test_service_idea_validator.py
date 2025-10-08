import unittest
from validators.service_idea_validator import validate_service_ideas

class TestServiceIdeaValidator(unittest.TestCase):

    def setUp(self):
        self.valid_idea = {
            "service_name": "AI Hygiene Consultant",
            "description": "A subscription service that automatically controls appliances.",
            "solved_pain_points": ["Anxiety about germs", "Repetitive chores"],
            "service_scalability": "Can be expanded to a premium paid model."
        }
        self.invalid_idea = {
            "service_name": "Incomplete Idea",
            "description": "This idea is missing required fields.",
            # Missing "solved_pain_points" and "service_scalability"
        }

    def test_validate_service_ideas_with_valid_data(self):
       
        print("\nRunning test: test_validate_service_ideas_with_valid_data")
        ideas_data = [self.valid_idea]
        result = validate_service_ideas(ideas_data)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['service_name'], self.valid_idea['service_name'])

    def test_validate_service_ideas_with_mixed_data(self):
       
        print("\nRunning test: test_validate_service_ideas_with_mixed_data")
        ideas_data = [self.valid_idea, self.invalid_idea]
        result = validate_service_ideas(ideas_data)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['service_name'], self.valid_idea['service_name'])

    def test_validate_service_ideas_with_non_list_input(self):
      
        print("\nRunning test: test_validate_service_ideas_with_non_list_input")
        result = validate_service_ideas("not a list")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

if __name__ == '__main__':
    unittest.main()