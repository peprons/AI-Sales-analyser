import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
from sales_analyzer import SalesAnalyzer
from pandasai import SmartDataframe # Make sure this import is added if not already there (it's used in SalesAnalyzer)
# Assuming SalesAnalyzer is directly importable from sales_analyzer.py
# and that pandasai, OpenAI, etc., are handled within SalesAnalyzer or mocked.

class TestSalesAnalyzerInit(unittest.TestCase):
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @patch('sales_analyzer.OpenAI')    # Mock OpenAI constructor
    @patch('sales_analyzer.pai.config.set')  # Mock pai.config.set from the sales_analyzer module context
    def test_init_success(self, mock_pai_config_set, mock_openai_constructor): # Order of args matters, outer patch first
        """Test successful initialization with API key."""
        # Setup mock return value for OpenAI constructor
        mock_llm_instance = MagicMock()
        mock_openai_constructor.return_value = mock_llm_instance

        analyzer = SalesAnalyzer()
        
        # Assertions
        mock_openai_constructor.assert_called_once_with(api_token="test_key")
        mock_pai_config_set.assert_called_once_with({"llm": mock_llm_instance})
        self.assertIs(analyzer.llm, mock_llm_instance, "analyzer.llm should be the mocked LLM instance")
        
        # To assert analyzer.pai is the pandasai module, we need to import pandasai in the test file
        # This import should already exist: import pandasai
        # The sales_analyzer.py does 'import pandasai as pai'. So analyzer.pai refers to the module.
        self.assertIsNotNone(analyzer.pai, "analyzer.pai should be set")
        import pandasai # Ensure pandasai is imported in the test file for this assertion
        self.assertIs(analyzer.pai, pandasai, "analyzer.pai should be the pandasai module itself")

    @patch.dict(os.environ, {}, clear=True) # Ensure OPENAI_API_KEY is not set
    @patch('sales_analyzer.OpenAI') # Mock OpenAI constructor
    def test_init_no_api_key(self, mock_openai_constructor): # Renamed arg for clarity
        """Test ValueError is raised if API key is missing."""
        with self.assertRaises(ValueError) as context:
            SalesAnalyzer()
        self.assertTrue('OpenAI API key not found' in str(context.exception))
        mock_openai_constructor.assert_not_called() # OpenAI should not be initialized

class TestSalesAnalyzerLoadData(unittest.TestCase):
    @patch('sales_analyzer.pai.config.set') # Mock pai.config.set
    @patch('sales_analyzer.OpenAI')         # Mock OpenAI constructor
    def setUp(self, mock_openai_constructor, mock_pai_config_set): # Order of args matters
        """Set up for test methods."""
        # Patch os.getenv to avoid needing a real API key for SalesAnalyzer instantiation
        self.env_patch = patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
        self.env_patch.start()
        
        # Configure mock for OpenAI constructor (already patched via decorator)
        self.mock_openai_instance = MagicMock()
        mock_openai_constructor.return_value = self.mock_openai_instance
        
        self.analyzer = SalesAnalyzer() # Now this will use the mocked pai.config.set
        self.test_csv_path = 'sample_test_data.csv'
        self.empty_csv_path = 'empty_test_data.csv'
        self.non_existent_csv_path = 'no_such_file.csv'

        # Create an empty CSV with headers for one of the tests
        with open(self.empty_csv_path, 'w') as f:
            f.write("Date,Product,Sales,Region\n") # Header row, matching sample_test_data.csv

    def tearDown(self):
        """Clean up after test methods."""
        self.env_patch.stop()
        # self.openai_patch.stop() # Removed as openai_patch is no longer stored on self
        if os.path.exists(self.empty_csv_path):
            os.remove(self.empty_csv_path)

    def test_load_data_success(self):
        """Test successful loading of a CSV file."""
        self.analyzer.load_data(self.test_csv_path)
        self.assertIsNotNone(self.analyzer.df)
        self.assertFalse(self.analyzer.df.empty)
        self.assertEqual(len(self.analyzer.df), 6) # Based on sample_test_data.csv

    def test_load_data_file_not_found(self):
        """Test FileNotFoundError for a non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.analyzer.load_data(self.non_existent_csv_path)

    def test_load_data_empty_csv(self):
        """Test loading an empty CSV file."""
        self.analyzer.load_data(self.empty_csv_path)
        self.assertIsNotNone(self.analyzer.df)
        self.assertTrue(self.analyzer.df.empty)

class TestSalesAnalyzerAnalyze(unittest.TestCase):
    @patch('sales_analyzer.pai.config.set') # Mock pai.config.set
    @patch('sales_analyzer.OpenAI')         # Mock OpenAI constructor
    def setUp(self, mock_openai_constructor, mock_pai_config_set): # Order of args matters
        """Set up for test methods."""
        self.env_patch = patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
        self.env_patch.start()
        
        # Configure mock for OpenAI constructor (already patched via decorator)
        self.mock_openai_instance = MagicMock()
        mock_openai_constructor.return_value = self.mock_openai_instance
        
        self.analyzer = SalesAnalyzer() # Now this will use the mocked pai.config.set
        self.test_csv_path = 'sample_test_data.csv'

    def tearDown(self):
        """Clean up after test methods."""
        self.env_patch.stop()
        # self.openai_patch.stop() # Removed as openai_patch is no longer stored on self

    @patch('sales_analyzer.SmartDataframe.chat') # Mock the chat method
    def test_analyze_success(self, mock_chat):
        """Test successful analysis with loaded data."""
        mock_chat.return_value = "Analysis result"
        self.analyzer.load_data(self.test_csv_path) # Load data first
        result = self.analyzer.analyze("Test query")
        self.assertEqual(result, "Analysis result")
        mock_chat.assert_called_once_with("Test query")

    def test_analyze_no_data_loaded(self):
        """Test ValueError if analyze is called before loading data."""
        with self.assertRaises(ValueError) as context:
            self.analyzer.analyze("Test query")
        self.assertTrue('No data loaded' in str(context.exception))

    @patch('sales_analyzer.SmartDataframe.chat') # Mock the chat method
    def test_analyze_exception_in_chat(self, mock_chat):
        """Test that None is returned if SmartDataframe.chat() raises an exception."""
        mock_chat.side_effect = Exception("LLM error")
        self.analyzer.load_data(self.test_csv_path) # Load data first
        # Patch print to suppress error output during test
        with patch('builtins.print') as mock_print:
            result = self.analyzer.analyze("Test query")
            self.assertIsNone(result)
            mock_print.assert_any_call('Error during analysis: LLM error')

# It's crucial to mock 'sales_analyzer.plt' if 'plt' is used as a module-level import in sales_analyzer.py
# or 'pandasai.helpers.matplotlib.plt' if that's where it's coming from,
# or more generally, the source where 'plt' is actually resolved.
# Given the current sales_analyzer.py, 'plt' is not defined.
# This implies 'plt' is expected to be in the global scope or imported by a dependency like pandasai.
# For robust mocking, we will assume 'plt' is available in the scope of save_plot.
# The most common way this happens is 'import matplotlib.pyplot as plt' somewhere.
# If sales_analyzer.py is run as a script and pandasai imports plt,
# then we might need to patch 'pandasai.plot_matplotlib.plt' or similar.
# Let's proceed by patching 'matplotlib.pyplot' as it's the most standard.
# If this fails, it means the 'plt' in save_plot comes from somewhere else.

@patch('sales_analyzer.plt', create=True) # create=True allows patching a non-existent attribute for the test
class TestSalesAnalyzerSavePlot(unittest.TestCase):
    @patch('sales_analyzer.pai.config.set') # Mock pai.config.set
    @patch('sales_analyzer.OpenAI')         # Mock OpenAI constructor
    def setUp(self, mock_openai_constructor, mock_pai_config_set): # Order of args matters
        """Set up for test methods."""
        self.env_patch = patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
        self.env_patch.start()
        
        # Configure mock for OpenAI constructor (already patched via decorator)
        self.mock_openai_instance = MagicMock()
        mock_openai_constructor.return_value = self.mock_openai_instance
        
        self.analyzer = SalesAnalyzer() # Now this will use the mocked pai.config.set

    def tearDown(self):
        """Clean up after test methods."""
        self.env_patch.stop()
        # self.openai_patch.stop() # Removed as openai_patch is no longer stored on self

    def test_save_plot_plot_exists(self, mock_plt):
        """Test save_plot when a plot exists (get_fignums returns non-empty)."""
        mock_plt.get_fignums.return_value = [1] # Simulate one open figure
        
        with patch('builtins.print') as mock_print:
            self.analyzer.save_plot("test_plot.png")
            mock_plt.savefig.assert_called_once_with("test_plot.png")
            mock_plt.close.assert_called_once()
            mock_print.assert_called_once_with('Plot saved as test_plot.png')

    def test_save_plot_no_plot_exists(self, mock_plt):
        """Test save_plot when no plot exists (get_fignums returns empty)."""
        mock_plt.get_fignums.return_value = [] # Simulate no open figures
        
        with patch('builtins.print') as mock_print:
            self.analyzer.save_plot("test_plot.png")
            mock_plt.savefig.assert_not_called()
            mock_plt.close.assert_not_called()
            mock_print.assert_not_called()

    def test_save_plot_custom_filename(self, mock_plt):
        """Test save_plot with a custom filename."""
        mock_plt.get_fignums.return_value = [1]
        custom_filename = "custom_analysis.jpg"
        with patch('builtins.print') as mock_print:
            self.analyzer.save_plot(filename=custom_filename)
            mock_plt.savefig.assert_called_once_with(custom_filename)
            mock_plt.close.assert_called_once()
            mock_print.assert_called_once_with(f'Plot saved as {custom_filename}')

if __name__ == '__main__':
    unittest.main()
