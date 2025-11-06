#!/bin/bash
# Quick SSH setup script for Jetson
# Run this on your Jetson device

set -e

echo "=========================================="
echo "SSH Setup for Jetson"
echo "=========================================="
echo ""

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then 
    echo "This script needs sudo privileges."
    echo "Please run: sudo bash setup_ssh_jetson.sh"
    exit 1
fi

echo "Step 1: Installing SSH server..."
apt-get update
apt-get install -y openssh-server

echo ""
echo "Step 2: Starting SSH service..."
systemctl enable ssh
systemctl start ssh

echo ""
echo "Step 3: Checking SSH status..."
systemctl status ssh --no-pager | head -n 5

echo ""
echo "Step 4: Finding IP address..."
IP_ADDRESS=$(hostname -I | awk '{print $1}')
echo "Your Jetson's IP address is: $IP_ADDRESS"

echo ""
echo "=========================================="
echo "SSH Setup Complete!"
echo "=========================================="
echo ""
echo "You can now connect from your Mac using:"
echo "  ssh acre@$IP_ADDRESS"
echo ""
echo "Or if you set up SSH keys:"
echo "  ssh jetson  (if configured in ~/.ssh/config)"
echo ""
echo "To transfer files:"
echo "  scp -r ~/Documents/acre acre@$IP_ADDRESS:/home/acre/ACRE_Capstone/"
echo ""

