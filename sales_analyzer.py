import os
from dotenv import load_dotenv
import pandasai as pai
from pandasai.llm import LLM
import pandas as pd
from pandasai_openai import OpenAI
from pandasai import SmartDataframe


class SalesAnalyzer:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI with API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError('OpenAI API key not found. Please set it in .env file')
        
        # Initialize PandasAI with OpenAI
        llm = OpenAI(api_token=api_key)
        pai.config.set({"llm": llm})
        self.llm = llm
        self.pai = pai
        self.df = None

    def load_data(self, file_path):
        """Load sales data from CSV file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'CSV file not found: {file_path}')
        print(f'Loading data with rows')
        self.df = pd.read_csv(file_path)

    def analyze(self, query):
        """Analyze sales data using natural language query"""
        if self.df is None:
            raise ValueError('No data loaded. Please load a CSV file first')
        smart_df = SmartDataframe(self.df, config={"llm": self.llm})
        try:
            result = smart_df.chat(query)
            return result
        except Exception as e:
            print(f'Error during analysis: {str(e)}')
            return None

    def save_plot(self, filename='analysis_plot.png'):
        """Save the last generated plot"""
        if plt.get_fignums():
            plt.savefig(filename)
            plt.close()
            print(f'Plot saved as {filename}')

def main():
    # Example usage
    analyzer = SalesAnalyzer()
    analyzer.load_data('sample_sales_data.csv')
    result = analyzer.analyze('What are the total sales by month?')
    print(result) 
    # Load sample data
    print('\nPlease provide the path to your CSV file when running the script.')
    print('Example usage:')
    print('1. Load data: analyzer.load_data("sales_data.csv")')
    print('2. Analyze: analyzer.analyze("What are the total sales by month?")')
    print('3. Save plot: analyzer.save_plot("sales_analysis.png")')

if __name__ == '__main__':
    main()