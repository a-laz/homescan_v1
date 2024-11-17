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

4. **Download SAM2 Checkpoint**
   ```bash
   # On macOS:
   curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
   
   # On Linux:
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
   
   # Or download directly in browser:
   # https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
   ```

5. **Set Up Environment Variables**
   ```bash
   # Create .env file
   touch .env
   
   # Add required environment variables to .env:
   # DATABASE_URL=your_database_url
   # SECRET_KEY=your_secret_key
   # Add any other required environment variables
   ```

6. **Initialize Database**
   ```bash
   python manage.py migrate
   ```

7. **Create Admin User** (Optional)
   ```bash
   python manage.py createsuperuser
   ```

8. **Run Development Server**
   ```bash
   python manage.py runserver
   ```

9. **Access from Other Devices** (Optional)
   To access the web app from other devices on your network (like your phone):

   1. Find your computer's IP address:
      ```bash
      # On Windows:
      ipconfig
      
      # On Mac/Linux:
      ifconfig
      # or
      ip addr
      ```
      Look for IPv4 address (usually starts with 192.168.x.x or 10.0.x.x)

   2. Update `ALLOWED_HOSTS` in `homescan/settings.py`:
      ```python
      ALLOWED_HOSTS = [
          'localhost',
          '127.0.0.1',
          '192.168.1.XXX',  # Replace with your computer's IP address
      ]
      ```

   3. Run the server with:
      ```bash
      python manage.py runserver 0.0.0.0:8000
      ```

   4. On your phone or other device:
      - Connect to the same WiFi network as your computer
      - Open web browser
      - Navigate to: `http://192.168.1.XXX:8000` (replace with your computer's IP)

### Accessing Camera on Mobile Devices

### Option 1: Using SSL Server (Recommended for testing)

1. Install django-sslserver:
   ```bash
   pip install django-sslserver
   ```

2. Add 'sslserver' to INSTALLED_APPS in settings.py:
   ```python
   INSTALLED_APPS = [
       ...
       'sslserver',
       ...
   ]
   ```

3. Run the SSL server:
   ```bash
   python manage.py runsslserver 0.0.0.0:8000
   ```

4. Access the site using HTTPS:
   ```
   https://192.168.1.XXX:8000
   ```
   Note: You'll need to accept the security warning about the self-signed certificate.

### Troubleshooting Camera Issues
- Make sure you're using HTTPS
- Accept any certificate warnings in your browser
- Grant camera permissions when prompted
- If using iOS 14+, ensure camera access is enabled in Settings > Privacy > Camera

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


