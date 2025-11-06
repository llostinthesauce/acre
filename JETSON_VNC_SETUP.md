# VNC Remote Desktop Setup for Jetson

## What is VNC?

VNC (Virtual Network Computing) lets you see and control your Jetson's desktop from your Mac, just like you're sitting in front of it.

## Quick Setup

### On Your Jetson

**Option 1: Use the setup script**
```bash
cd ~/ACRE_Capstone/acre
sudo bash setup_vnc_jetson.sh
```

**Option 2: Manual setup**
```bash
# Install VNC server
sudo apt-get update
sudo apt-get install -y tigervnc-standalone-server

# Set VNC password (you'll be prompted)
vncpasswd

# Start VNC server (display :1, port 5901)
vncserver :1 -geometry 1920x1080 -depth 24
```

### On Your Mac

**Option 1: Built-in Screen Sharing (Easiest)**

1. Open Finder
2. Press `Cmd+K` (or Go → Connect to Server)
3. Enter: `vnc://<JETSON_IP>:5901`
   - Example: `vnc://192.168.1.100:5901`
4. Click Connect
5. Enter the VNC password you set on the Jetson

**Option 2: Use a VNC Client**

Popular options:
- **RealVNC Viewer** (free): https://www.realvnc.com/download/viewer/
- **TigerVNC** (free): https://github.com/TigerVNC/tigervnc/releases
- **Jump Desktop** (paid, but excellent): https://jumpdesktop.com/

## VNC Display Numbers

- Display `:1` = Port `5901`
- Display `:2` = Port `5902`
- Display `:3` = Port `5903`
- etc.

## Managing VNC Server

```bash
# Start VNC server
vncserver :1

# Stop VNC server
vncserver -kill :1

# List running VNC servers
vncserver -list

# Start with specific resolution
vncserver :1 -geometry 1920x1080 -depth 24
```

## Auto-Start VNC on Boot

If you used the setup script, VNC will start automatically. Otherwise:

```bash
# Create systemd service (see setup_vnc_jetson.sh for details)
sudo systemctl enable vncserver@1.service
sudo systemctl start vncserver@1.service
```

## Troubleshooting

### Can't Connect

1. **Check if VNC is running:**
   ```bash
   vncserver -list
   ```

2. **Check firewall:**
   ```bash
   sudo ufw allow 5901
   ```

3. **Check if port is listening:**
   ```bash
   netstat -tlnp | grep 5901
   ```

### Black Screen / No Desktop

Edit `~/.vnc/xstartup`:
```bash
nano ~/.vnc/xstartup
```

Make sure it contains:
```bash
#!/bin/bash
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
[ -x /etc/vnc/xstartup ] && exec /etc/vnc/xstartup
[ -r $HOME/.Xresources ] && xrdb $HOME/.Xresources
x-window-manager &
startxfce4 &  # or gnome-session, startlxde, etc.
```

Make it executable:
```bash
chmod +x ~/.vnc/xstartup
```

Then restart VNC:
```bash
vncserver -kill :1
vncserver :1
```

### Performance Issues

- Lower the resolution: `vncserver :1 -geometry 1280x720`
- Reduce color depth: `vncserver :1 -depth 16`
- Use a wired connection instead of WiFi

## Alternative: X11 Forwarding (SSH Only)

If you just need to run GUI apps over SSH (not full desktop):

```bash
# From Mac (requires XQuartz)
ssh -X acre@<JETSON_IP>
# Then run GUI apps - they'll display on your Mac
```

Install XQuartz on Mac: https://www.xquartz.org/

## Alternative: NoMachine (Easier Setup)

NoMachine is easier to set up and often faster:

1. **On Jetson:**
   ```bash
   wget https://download.nomachine.com/download/8.x/Linux/nomachine_8.x.x_x86_64.deb
   sudo dpkg -i nomachine_*.deb
   ```

2. **On Mac:**
   - Download NoMachine client: https://www.nomachine.com/download

3. **Connect:**
   - Open NoMachine on Mac
   - Enter Jetson IP address
   - Login with your Jetson credentials

## Security Note

VNC passwords are not encrypted. For better security:
- Only use on trusted networks
- Consider SSH tunneling (see below)
- Or use NoMachine which has better encryption

## SSH Tunnel (More Secure)

Tunnel VNC through SSH for encryption:

```bash
# On Mac, create SSH tunnel
ssh -L 5901:localhost:5901 acre@<JETSON_IP>

# Then connect VNC to: localhost:5901
```

## Quick Reference

```bash
# On Jetson - Start VNC
vncserver :1

# On Jetson - Stop VNC
vncserver -kill :1

# On Mac - Connect
# Finder → Cmd+K → vnc://<JETSON_IP>:5901
```

