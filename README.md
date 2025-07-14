# rs-fastsam
Zero-shot satellite image segmentation using lightweight "Segment Anything" models.

## Setup

The default setup in this repository includes analysis images and a URL to a tile cache to visualise the imagery in Streamlit. 

`data/` includes two copies of a Sentinel 2 image from 16 May 2025. An 8-bit RGB version is used for the SAM-type segmentation. A downscaled 16 bit version including NIR, SWIR, red-edge bands is used for the post-segmentation land use classification.

### Prerequisites

- Python 3.12+
- Docker (optional, for containerized deployment)

### Without Docker

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/rs-fastsam.git
   cd rs-fastsam
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the application**
   - Copy and modify the configuration file as needed:
   ```bash
   cp config.yaml config.local.yaml  # Optional: create local config
   ```
   - Update paths in `config.yaml` to match your local setup

4. **Download required models**
   - The app will automatically download models on first run
   - Or manually place model files in the `models/` directory as specified in `config.yaml`

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

   - The app will be available at `http://localhost:8501`

### Using Docker

1. **Build the Docker image**
   ```bash
   docker build -t rs-fastsam .
   ```

2. **Run the container**
   ```bash
   docker run -p 8501:8080 rs-fastsam
   ```

### Configuration

Edit `config.yaml` to change:
- Analysis image paths
- Map settings and tile cache url
- Default model parameters
- UI model parameter options