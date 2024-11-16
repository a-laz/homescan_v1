# HomeScan

## Installation Guide

### Prerequisites

- Python 3.11 or higher
- pip (Python package installer)
- Git

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/a-laz/homescan_v1.git
   cd homescan_v1
   ```

2. **Set Up Virtual Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   ```bash
   # Create .env file
   touch .env
   
   # Add required environment variables to .env:
   # DATABASE_URL=your_database_url
   # SECRET_KEY=your_secret_key
   # Add any other required environment variables
   ```

5. **Initialize Database**
   ```bash
   python manage.py migrate
   ```

6. **Create Admin User** (Optional)
   ```bash
   python manage.py createsuperuser
   ```

7. **Run Development Server**
   ```bash
   python manage.py runserver
   ```

### Troubleshooting

- If you encounter any package installation errors, try:
  ```bash
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

- For database connection issues:
  - Verify your database credentials in .env
  - Ensure your database service is running

### System Requirements

- Memory: 4GB RAM minimum
- Storage: 1GB free space
- OS: macOS, Linux, or Windows


