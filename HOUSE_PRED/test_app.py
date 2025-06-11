import unittest
import pandas as pd
from app import preprocess_input # Assuming app.py is in the same directory or accessible

class TestPreprocessInput(unittest.TestCase):

    def test_preprocess_function(self):
        # Sample input data
        medInc = 8.3252
        houseAge = 41.0
        numRooms = 6.98412698
        numBedrooms = 1.02380952
        population = 322.0
        latitude = 37.88
        longitude = -122.23

        # Expected column names
        expected_columns = ['MedInc', 'HouseAge', 'NumRooms', 'NumBedrooms', 'Population', 'Latitude', 'Longitude']

        # Call the function
        result_df = preprocess_input(medInc, houseAge, numRooms, numBedrooms, population, latitude, longitude)

        # 1. Check if the result is a Pandas DataFrame
        self.assertIsInstance(result_df, pd.DataFrame, "Output is not a Pandas DataFrame")

        # 2. Check if the DataFrame has the correct column names
        self.assertListEqual(list(result_df.columns), expected_columns, "DataFrame columns are not correct")

        # 3. Check if the DataFrame has one row
        self.assertEqual(len(result_df), 1, "DataFrame should have one row")

        # 4. Check if the values in the DataFrame are correct
        expected_values = [medInc, houseAge, numRooms, numBedrooms, population, latitude, longitude]
        actual_values = result_df.iloc[0].tolist()

        for expected, actual in zip(expected_values, actual_values):
            self.assertEqual(expected, actual, f"Mismatch in data: expected {expected}, got {actual}")

if __name__ == '__main__':
    # This allows running the tests directly
    # However, for discoverability, it's better to run with `python -m unittest discover` or `python -m unittest test_app.py`
    # For simplicity in this environment, we might need to adjust how 'app' is imported if test_app.py is not in the same dir as app.py
    # For now, assuming they are effectively in the same directory for the import to work.
    unittest.main()
