# Data Processing and Data Structure

## Data Acquisition

The data is obtained through web scraping of university application FAQ pages. The following steps outline the data acquisition process:

1. **Web Scraping**: The application scrapes the FAQ pages of various universities to collect data.
2. **Data Transformation**: The scraped data is transformed into Excel columns with the following structure:
   - **Tags**: Store the university, department, and school information.
   - **Question and Answer**: Store the text of the questions and answers.
   - **Link**: Store the URL of the FAQ page.

## Data Processing

I read through all the excel files under the `/data` folder and turn them in to `Documents` to store in the **FAISS** vector database.

1. **File Loading**: The application scans the directory for Excel files (`.xlsx` or `.xls`).
2. **Data Extraction**: Each file is read into a DataFrame using `pandas`.
3. **Row Processing**: Each row in the DataFrame is processed to extract relevant information:
   - **Tags**: Extracted and split to determine the school and department.
   - **Question and Answer**: Text fields are cleaned and formatted.
4. **Document Creation**: A JSON object is created for each row, containing the question and answer, along with metadata such as school, department, and link.
5. **Document Storage**: The JSON object is converted to a string and stored in a `Document` object, which is then added to a list of documents.

## Data Structure

The data structure used in the application consists of the following components:

- **Document**: Represents a single document with the following attributes:
  - `page_content`: A JSON string containing the question and answer, along with metadata such as school, department, and link.
