import unittest
from validators.final_document_validator import validate_final_document

class TestFinalDocumentValidator(unittest.TestCase):
    """Unit tests for the final document validator."""

    def setUp(self):
        """Set up a valid final document structure for tests."""
        self.valid_doc = {
            "title": "Final Report",
            "customer_delight_goal": "A delightful goal",
            "cx": {
                "target_definition": {"description": "desc", "quote": "quote", "market_info": "info"},
                "core_experience": {"title": "title", "care": "care", "customization": ["cust"], "servitization": "serv"}
            },
            "performance": {
                "concept": {"find": "find", "unique": ["unique"]}
            },
            "dx": {
                "trigger": {"title": "title_t", "items": ["item_t"]},
                "accelerator": {"title": "title_a", "up_contents_service": ["up"], "data_driven_experience": ["dde"]},
                "tracker": {"title": "title_tr", "items": ["item_tr"]}
            }
        }

    def test_validate_final_document_with_valid_data(self):
        """Tests that a valid document passes validation."""
        print("\nRunning test: test_validate_final_document_with_valid_data")
        result = validate_final_document(self.valid_doc)
        self.assertIsNotNone(result)
        self.assertEqual(result['title'], "Final Report")

    def test_validate_final_document_with_invalid_data(self):
        """Tests that a document with a missing nested key fails validation."""
        print("\nRunning test: test_validate_final_document_with_invalid_data")
        # Create a deep copy to avoid modifying the original setUp data
        import copy
        invalid_doc = copy.deepcopy(self.valid_doc)
        
        # Invalidate by removing a required nested structure
        del invalid_doc["dx"]["trigger"]
        
        result = validate_final_document(invalid_doc)
        self.assertIsNone(result)
        
    def test_validate_final_document_with_non_dict_input(self):
        """Tests that a non-dictionary input returns None."""
        print("\nRunning test: test_validate_final_document_with_non_dict_input")
        result = validate_final_document(None)
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()