#!/bin/bash
# VNC Server setup for Jetson
# This allows you to see the GUI from your Mac

set -e

echo "=========================================="
echo "VNC Server Setup for Jetson"
echo "=========================================="
echo ""

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then 
    echo "This script needs sudo privileges."
    echo "Please run: sudo bash setup_vnc_jetson.sh"
    exit 1
fi

echo "Step 1: Installing VNC server..."
apt-get update
apt-get install -y tigervnc-standalone-server tigervnc-common

echo ""
echo "Step 2: Setting up VNC for user..."
USER=$(logname)
echo "Setting up VNC for user: $USER"

# Create VNC directory
mkdir -p /home/$USER/.vnc
chown $USER:$USER /home/$USER/.vnc

echo ""
echo "Step 3: Setting VNC password..."
echo "You'll be prompted to set a VNC password (different from your login password)"
echo "This password is used to connect from your Mac"
echo ""
su - $USER -c "vncpasswd"

echo ""
echo "Step 4: Creating VNC startup script..."
cat > /home/$USER/.vnc/xstartup << 'EOF'
#!/bin/bash
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
[ -x /etc/vnc/xstartup ] && exec /etc/vnc/xstartup
[ -r $HOME/.Xresources ] && xrdb $HOME/.Xresources
x-window-manager &
# Start your desktop environment
if [ -f /usr/bin/startxfce4 ]; then
    startxfce4 &
elif [ -f /usr/bin/gnome-session ]; then
    gnome-session &
elif [ -f /usr/bin/startlxde ]; then
    startlxde &
else
    # Fallback to basic window manager
    xterm &
fi
EOF

chmod +x /home/$USER/.vnc/xstartup
chown $USER:$USER /home/$USER/.vnc/xstartup

echo ""
echo "Step 5: Creating VNC service..."
cat > /etc/systemd/system/vncserver@.service << EOF
[Unit]
Description=Start TightVNC server at startup
After=syslog.target network.target

[Service]
Type=forking
User=$USER
Group=$USER
WorkingDirectory=/home/$USER

ExecStartPre=/bin/sh -c '/usr/bin/vncserver -kill :%i > /dev/null 2>&1 || :'
ExecStart=/usr/bin/vncserver -depth 24 -geometry 1920x1080 :%i
ExecStop=/usr/bin/vncserver -kill :%i

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable vncserver@1.service

echo ""
echo "Step 6: Starting VNC server..."
systemctl start vncserver@1.service

echo ""
echo "=========================================="
echo "VNC Setup Complete!"
echo "=========================================="
echo ""
IP_ADDRESS=$(hostname -I | awk '{print $1}')
echo "VNC Server is running on: $IP_ADDRESS:5901"
echo ""
echo "To connect from your Mac:"
echo "1. Install a VNC client (like 'Screen Sharing' built into Mac)"
echo "2. Connect to: $IP_ADDRESS:5901"
echo "   Or use: vnc://$IP_ADDRESS:5901"
echo ""
echo "VNC Display Number: 1 (port 5901)"
echo ""
echo "To stop VNC: sudo systemctl stop vncserver@1"
echo "To start VNC: sudo systemctl start vncserver@1"
echo "To restart VNC: sudo systemctl restart vncserver@1"
echo ""

