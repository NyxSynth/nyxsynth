#!/bin/bash
#
# NyxSynth One-Click Deployment Script
# This script automates the complete setup process for NyxSynth
#

# Text formatting
BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}"
echo "███╗   ██╗██╗   ██╗██╗  ██╗███████╗██╗   ██╗███╗   ██╗████████╗██╗  ██╗"
echo "████╗  ██║╚██╗ ██╔╝╚██╗██╔╝██╔════╝╚██╗ ██╔╝████╗  ██║╚══██╔══╝██║  ██║"
echo "██╔██╗ ██║ ╚████╔╝  ╚███╔╝ ███████╗ ╚████╔╝ ██╔██╗ ██║   ██║   ███████║"
echo "██║╚██╗██║  ╚██╔╝   ██╔██╗ ╚════██║  ╚██╔╝  ██║╚██╗██║   ██║   ██╔══██║"
echo "██║ ╚████║   ██║   ██╔╝ ██╗███████║   ██║   ██║ ╚████║   ██║   ██║  ██║"
echo "╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝"
echo -e "${NC}"

echo -e "${BOLD}Biomimetic Neural Cryptocurrency${NC}"
echo "One-Click Deployment Script"
echo "========================================"
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to display progress
display_progress() {
    local message="$1"
    local sleep_duration=${2:-0.1}
    echo -n -e "${message}... "
    for i in {1..3}; do
        echo -n "."
        sleep $sleep_duration
    done
    echo -e " ${GREEN}Done!${NC}"
}

# Function to show step header
show_step() {
    echo ""
    echo -e "${BLUE}[STEP $1] $2${NC}"
    echo "----------------------------------------"
}

# Check if running with proper permissions
if [ "$EUID" -ne 0 ]; then
    echo "This script requires administrative privileges."
    echo "Please run with sudo or as root."
    exit 1
fi

# Initialize variables
INSTALL_DIR="/opt/nyxsynth"
CONFIG_DIR="/etc/nyxsynth"
DATA_DIR="/var/lib/nyxsynth"
DEPLOY_TYPE="production"
START_ON_BOOT=true
CREATE_ADMIN=true
ADMIN_USER="admin"
ADMIN_PASS="nyxadmin123"

# Step 1: Check environment and prerequisites
show_step "1" "Checking system environment"

# Check OS
if command_exists lsb_release; then
    OS=$(lsb_release -si)
    VER=$(lsb_release -sr)
    echo "Operating System: $OS $VER"
else
    echo "Operating System: Unknown"
fi

# Check system resources
CPU_CORES=$(grep -c ^processor /proc/cpuinfo)
TOTAL_MEM=$(free -h | awk '/^Mem:/{print $2}')
AVAIL_DISK=$(df -h "$PWD" | awk 'NR==2 {print $4}')

echo "CPU Cores: $CPU_CORES"
echo "Memory: $TOTAL_MEM"
echo "Available Disk Space: $AVAIL_DISK"

# Check for prerequisites
echo "Checking prerequisites..."

MISSING_PREREQS=false

# Check Python
if command_exists python3; then
    PYTHON_VER=$(python3 --version | cut -d " " -f 2)
    echo -e "✅ Python found: ${GREEN}$PYTHON_VER${NC}"
    
    # Check Python version
    if [[ "$(echo "$PYTHON_VER" | cut -d. -f1)" -lt 3 ]] || [[ "$(echo "$PYTHON_VER" | cut -d. -f2)" -lt 8 ]]; then
        echo -e "❌ ${RED}Python 3.8+ required, found $PYTHON_VER${NC}"
        MISSING_PREREQS=true
    fi
else
    echo -e "❌ Python 3.8+ not found"
    MISSING_PREREQS=true
fi

# Check Node.js
if command_exists node; then
    NODE_VER=$(node --version)
    echo -e "✅ Node.js found: ${GREEN}$NODE_VER${NC}"
    
    # Check Node.js version
    if [[ "$(echo "$NODE_VER" | cut -d. -f1 | cut -c 2-)" -lt 14 ]]; then
        echo -e "❌ Node.js 14+ required, found $NODE_VER"
        MISSING_PREREQS=true
    fi
else
    echo -e "❌ Node.js 14+ not found"
    MISSING_PREREQS=true
fi

# Check Docker (for production deployment)
if command_exists docker && command_exists docker-compose; then
    DOCKER_VER=$(docker --version | cut -d " " -f 3 | cut -d "," -f 1)
    COMPOSE_VER=$(docker-compose --version | cut -d " " -f 3 | tr -d ",")
    echo -e "✅ Docker found: ${GREEN}$DOCKER_VER${NC}"
    echo -e "✅ Docker Compose found: ${GREEN}$COMPOSE_VER${NC}"
    HAS_DOCKER=true
else
    echo -e "⚠️ Docker and Docker Compose not found (required for production deployment)"
    HAS_DOCKER=false
fi

if [ "$MISSING_PREREQS" = true ]; then
    echo "Missing prerequisites. Please install them before continuing."
    echo "Would you like this script to install the missing prerequisites? (y/n)"
    read -r INSTALL_PREREQS
    
    if [[ "$INSTALL_PREREQS" =~ ^[Yy]$ ]]; then
        show_step "1.1" "Installing prerequisites"
        
        # Update package lists
        apt-get update
        
        # Install Python if needed
        if ! command_exists python3 || [[ "$(python3 --version | cut -d " " -f 2 | cut -d. -f1)" -lt 3 ]] || [[ "$(python3 --version | cut -d " " -f 2 | cut -d. -f2)" -lt 8 ]]; then
            echo "Installing Python 3.8+..."
            apt-get install -y python3 python3-pip python3-venv
        fi
        
        # Install Node.js if needed
        if ! command_exists node || [[ "$(node --version | cut -d. -f1 | cut -c 2-)" -lt 14 ]]; then
            echo "Installing Node.js 14+..."
            curl -fsSL https://deb.nodesource.com/setup_16.x | bash -
            apt-get install -y nodejs
        fi
        
        # Install Docker if needed for production
        if [[ "$DEPLOY_TYPE" == "production" ]] && [ "$HAS_DOCKER" = false ]; then
            echo "Installing Docker and Docker Compose..."
            curl -fsSL https://get.docker.com | sh
            apt-get install -y docker-compose
        fi
        
        echo -e "${GREEN}Prerequisites installed successfully!${NC}"
    else
        echo "Please install the missing prerequisites and run this script again."
        exit 1
    fi
fi

# Step 2: Configure installation
show_step "2" "Configuration"

# Ask for installation type
echo "Select deployment type:"
echo "1) Development (for testing and development)"
echo "2) Production (for live environments, uses Docker)"
read -r DEPLOY_TYPE_CHOICE

if [[ "$DEPLOY_TYPE_CHOICE" == "1" ]]; then
    DEPLOY_TYPE="development"
else
    DEPLOY_TYPE="production"
    
    if [ "$HAS_DOCKER" = false ]; then
        echo "Docker is required for production deployment. Please install Docker and Docker Compose first."
        exit 1
    fi
fi

# Ask for installation directory
echo "Installation directory [$INSTALL_DIR]:"
read -r CUSTOM_INSTALL_DIR
if [[ -n "$CUSTOM_INSTALL_DIR" ]]; then
    INSTALL_DIR="$CUSTOM_INSTALL_DIR"
fi

# Create admin user
echo "Create admin user? (y/n) [y]:"
read -r CREATE_ADMIN_CHOICE
if [[ "$CREATE_ADMIN_CHOICE" =~ ^[Nn]$ ]]; then
    CREATE_ADMIN=false
else
    echo "Admin username [$ADMIN_USER]:"
    read -r CUSTOM_ADMIN_USER
    if [[ -n "$CUSTOM_ADMIN_USER" ]]; then
        ADMIN_USER="$CUSTOM_ADMIN_USER"
    fi
    
    echo "Admin password (leave blank to generate):"
    read -r -s CUSTOM_ADMIN_PASS
    if [[ -n "$CUSTOM_ADMIN_PASS" ]]; then
        ADMIN_PASS="$CUSTOM_ADMIN_PASS"
    else
        # Generate random password
        ADMIN_PASS=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 16 | head -n 1)
    fi
    echo
fi

# Start on boot
echo "Start NyxSynth on system boot? (y/n) [y]:"
read -r START_ON_BOOT_CHOICE
if [[ "$START_ON_BOOT_CHOICE" =~ ^[Nn]$ ]]; then
    START_ON_BOOT=false
fi

# Step 3: Download and prepare NyxSynth
show_step "3" "Downloading NyxSynth"

# Create installation directory
mkdir -p "$INSTALL_DIR"
mkdir -p "$CONFIG_DIR"
mkdir -p "$DATA_DIR"

# Clone repository
echo "Cloning NyxSynth repository..."
git clone https://github.com/yourusername/nyxsynth.git "$INSTALL_DIR/source"
cd "$INSTALL_DIR/source"

# Step 4: Installation
show_step "4" "Installing NyxSynth"

if [[ "$DEPLOY_TYPE" == "development" ]]; then
    # Development installation
    echo "Performing development installation..."
    
    # Create Python virtual environment
    display_progress "Creating Python virtual environment"
    python3 -m venv "$INSTALL_DIR/venv"
    source "$INSTALL_DIR/venv/bin/activate"
    
    # Install Python dependencies
    display_progress "Installing Python dependencies"
    pip install -r requirements.txt
    
    # Install frontend dependencies
    display_progress "Installing frontend dependencies"
    cd "$INSTALL_DIR/source/frontend"
    npm install
    cd "$INSTALL_DIR/source/frontend/admin"
    npm install
    
    # Return to source directory
    cd "$INSTALL_DIR/source"
    
    # Initialize blockchain
    display_progress "Initializing blockchain"
    python3 scripts/initialize.py --config-dir="$CONFIG_DIR" --data-dir="$DATA_DIR" \
                                --admin-user="$ADMIN_USER" --admin-pass="$ADMIN_PASS"
    
    # Create startup script
    cat > "$INSTALL_DIR/start.sh" << 'EOL'
#!/bin/bash
source "$(dirname "$0")/venv/bin/activate"
cd "$(dirname "$0")/source"
python3 api/server.py &
SERVER_PID=$!
cd frontend && npm start &
FRONTEND_PID=$!
trap "kill $SERVER_PID $FRONTEND_PID; exit" INT TERM
wait
EOL
    chmod +x "$INSTALL_DIR/start.sh"
    
    # Create service file for systemd if requested
    if [ "$START_ON_BOOT" = true ]; then
        cat > /etc/systemd/system/nyxsynth.service << EOL
[Unit]
Description=NyxSynth Cryptocurrency
After=network.target

[Service]
ExecStart=$INSTALL_DIR/start.sh
WorkingDirectory=$INSTALL_DIR
User=root
Group=root
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOL
        systemctl daemon-reload
        systemctl enable nyxsynth.service
    fi
    
else
    # Production installation
    echo "Performing production installation..."
    
    # Create Docker Compose configuration
    display_progress "Configuring Docker environment"
    
    # Create environment file
    cat > "$INSTALL_DIR/.env" << EOL
NYXSYNTH_CONFIG_DIR=$CONFIG_DIR
NYXSYNTH_DATA_DIR=$DATA_DIR
ADMIN_USER=$ADMIN_USER
ADMIN_PASS=$ADMIN_PASS
EOL
    
    # Build Docker containers
    display_progress "Building Docker containers"
    docker-compose -f "$INSTALL_DIR/source/docker-compose.yml" build
    
    # Initialize the blockchain
    display_progress "Initializing blockchain"
    docker-compose -f "$INSTALL_DIR/source/docker-compose.yml" run --rm nyxsynth-backend python scripts/initialize.py \
                   --config-dir="/etc/nyxsynth" --data-dir="/var/lib/nyxsynth" \
                   --admin-user="$ADMIN_USER" --admin-pass="$ADMIN_PASS"
    
    # Create startup script
    cat > "$INSTALL_DIR/start.sh" << 'EOL'
#!/bin/bash
cd "$(dirname "$0")/source"
docker-compose up -d
EOL
    chmod +x "$INSTALL_DIR/start.sh"
    
    # Create shutdown script
    cat > "$INSTALL_DIR/stop.sh" << 'EOL'
#!/bin/bash
cd "$(dirname "$0")/source"
docker-compose down
EOL
    chmod +x "$INSTALL_DIR/stop.sh"
    
    # Create service file for systemd if requested
    if [ "$START_ON_BOOT" = true ]; then
        cat > /etc/systemd/system/nyxsynth.service << EOL
[Unit]
Description=NyxSynth Cryptocurrency
After=docker.service
Requires=docker.service

[Service]
ExecStart=$INSTALL_DIR/start.sh
ExecStop=$INSTALL_DIR/stop.sh
WorkingDirectory=$INSTALL_DIR
Type=forking
User=root
Group=root
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOL
        systemctl daemon-reload
        systemctl enable nyxsynth.service
    fi
fi

# Step 5: Start NyxSynth
show_step "5" "Starting NyxSynth"

if [ "$START_ON_BOOT" = true ]; then
    display_progress "Starting NyxSynth service"
    systemctl start nyxsynth.service
else
    echo "Starting NyxSynth manually..."
    "$INSTALL_DIR/start.sh"
fi

# Step 6: Installation complete
show_step "6" "Installation Complete"

echo -e "${GREEN}NyxSynth has been successfully installed!${NC}"
echo ""
echo "Installation Details:"
echo "====================="
echo "Installation Directory: $INSTALL_DIR"
echo "Configuration Directory: $CONFIG_DIR"
echo "Data Directory: $DATA_DIR"
echo "Deployment Type: $DEPLOY_TYPE"

if [ "$CREATE_ADMIN" = true ]; then
    echo ""
    echo "Admin Credentials:"
    echo "=================="
    echo "Username: $ADMIN_USER"
    echo "Password: $ADMIN_PASS"
    echo ""
    echo -e "${BOLD}IMPORTANT: Save these credentials securely!${NC}"
fi

echo ""
echo "Access Information:"
echo "=================="

if [[ "$DEPLOY_TYPE" == "development" ]]; then
    echo "Frontend URL: http://localhost:3000"
    echo "API URL: http://localhost:5000"
    echo "Admin Dashboard: http://localhost:3000/admin"
else
    # Get server IP address
    SERVER_IP=$(hostname -I | awk '{print $1}')
    echo "Frontend URL: http://$SERVER_IP"
    echo "API URL: http://$SERVER_IP/api"
    echo "Admin Dashboard: http://$SERVER_IP/admin"
fi

echo ""
echo "Management Commands:"
echo "===================="
echo "Start NyxSynth: $INSTALL_DIR/start.sh"

if [[ "$DEPLOY_TYPE" == "production" ]]; then
    echo "Stop NyxSynth: $INSTALL_DIR/stop.sh"
fi

if [ "$START_ON_BOOT" = true ]; then
    echo "Service control: systemctl [start|stop|restart|status] nyxsynth"
fi

echo ""
echo -e "${PURPLE}Thank you for choosing NyxSynth!${NC}"
echo -e "The biomimetic neural cryptocurrency of the future."
echo -e "Visit https://nyxsynth.com for documentation and support."
echo ""