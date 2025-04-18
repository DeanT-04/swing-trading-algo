# Installation Guide

This guide explains how to install and set up the Swing Trading Algorithm.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)

## Installation Steps

### 1. Clone the Repository (or Download)

```bash
git clone https://github.com/yourusername/swing-trading-algo.git
cd swing-trading-algo
```

Alternatively, download and extract the ZIP file from the GitHub repository.

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to avoid conflicts with other Python packages.

#### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Application

Copy the template configuration file and edit it with your settings:

```bash
cp config/config_template.yaml config/config.yaml
```

Edit `config/config.yaml` with your preferred text editor to set your API keys, trading parameters, and other settings.

### 5. Verify Installation

Run the tests to verify that everything is working correctly:

```bash
python run_tests.py
```

## Data Provider Setup

### Alpha Vantage

If you're using Alpha Vantage as your data provider:

1. Sign up for a free API key at [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Add your API key to the `config/config.yaml` file:

```yaml
data:
  provider: "alpha_vantage"
  api_key: "YOUR_API_KEY_HERE"
  # Other data settings...
```

### CSV Data

If you're using CSV files as your data provider:

1. Create a directory for your CSV files (e.g., `data/csv`)
2. Add your CSV files to this directory with the naming format `SYMBOL_TIMEFRAME.csv` (e.g., `AAPL_daily.csv`)
3. Update your `config/config.yaml` file:

```yaml
data:
  provider: "csv"
  data_dir: "data/csv"
  # Other data settings...
```

## Troubleshooting

### Common Issues

1. **ImportError**: Make sure you're running the application from the project root directory and that your virtual environment is activated.

2. **API Key Issues**: Double-check that your API key is correctly entered in the configuration file.

3. **Missing Dependencies**: If you encounter missing dependencies, try reinstalling the requirements:

   ```bash
   pip install -r requirements.txt
   ```

4. **Permission Issues**: If you encounter permission issues when creating directories or files, check that your user has write permissions to the project directory.

### Getting Help

If you encounter any issues not covered here, please open an issue on the GitHub repository or contact the project maintainers.
