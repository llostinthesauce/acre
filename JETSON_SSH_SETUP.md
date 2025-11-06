# SSH Setup for Jetson

## On Your Jetson Device

### 1. Install SSH Server (if not already installed)

```bash
sudo apt-get update
sudo apt-get install -y openssh-server
```

### 2. Start SSH Service

```bash
sudo systemctl enable ssh
sudo systemctl start ssh
```

### 3. Check SSH Status

```bash
sudo systemctl status ssh
```

You should see "active (running)".

### 4. Find Your Jetson's IP Address

```bash
# Method 1: Using hostname
hostname -I

# Method 2: Using ip command
ip addr show | grep "inet " | grep -v 127.0.0.1

# Method 3: Using ifconfig (if installed)
ifconfig | grep "inet " | grep -v 127.0.0.1
```

You'll see something like `192.168.1.100` - this is your Jetson's IP address.

### 5. (Optional) Set Static IP (Recommended)

If your Jetson's IP keeps changing, set a static IP:

```bash
sudo nano /etc/netplan/01-netcfg.yaml
```

Add/edit to something like:
```yaml
network:
  version: 2
  renderer: networkd
  ethernet:
    eth0:
      dhcp4: no
      addresses:
        - 192.168.1.100/24
      gateway4: 192.168.1.1
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]
```

Then apply:
```bash
sudo netplan apply
```

**Note:** Adjust the IP address, gateway, and interface name (eth0) to match your network.

## On Your Mac

### 1. Connect via SSH

```bash
ssh acre@<JETSON_IP_ADDRESS>
```

Replace `<JETSON_IP_ADDRESS>` with the IP you found in step 4.

For example:
```bash
ssh acre@192.168.1.100
```

### 2. First Time Connection

You'll see a message about host authenticity - type `yes` to continue.

### 3. Enter Password

Enter the password for the `acre` user on your Jetson.

### 4. (Optional) Set Up SSH Keys (Passwordless Login)

On your Mac, generate an SSH key (if you don't have one):

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Press Enter to accept default location. You can set a passphrase or leave it empty.

Copy your public key to the Jetson:

```bash
ssh-copy-id acre@<JETSON_IP_ADDRESS>
```

Now you can SSH without entering a password each time!

### 5. (Optional) Create SSH Config for Easy Access

On your Mac, edit or create `~/.ssh/config`:

```bash
nano ~/.ssh/config
```

Add:
```
Host jetson
    HostName <JETSON_IP_ADDRESS>
    User acre
    Port 22
```

Now you can just type:
```bash
ssh jetson
```

## Transferring Files

### From Mac to Jetson

```bash
# Copy a file
scp /path/to/file acre@<JETSON_IP>:/home/acre/

# Copy a directory
scp -r /path/to/directory acre@<JETSON_IP>:/home/acre/

# Example: Copy the acre project
scp -r ~/Documents/acre acre@192.168.1.100:/home/acre/ACRE_Capstone/
```

### From Jetson to Mac

```bash
# Copy a file
scp acre@<JETSON_IP>:/path/to/file ~/Downloads/

# Copy a directory
scp -r acre@<JETSON_IP>:/path/to/directory ~/Downloads/
```

## Troubleshooting

### Can't Connect

1. **Check if SSH is running on Jetson:**
   ```bash
   sudo systemctl status ssh
   ```

2. **Check firewall (if enabled):**
   ```bash
   sudo ufw status
   sudo ufw allow ssh  # If firewall is blocking
   ```

3. **Check if you're on the same network:**
   - Both devices must be on the same WiFi/LAN network

4. **Ping the Jetson:**
   ```bash
   ping <JETSON_IP_ADDRESS>
   ```

### Connection Refused

- Make sure SSH server is installed and running
- Check that port 22 is not blocked by firewall
- Verify the IP address is correct

### Permission Denied

- Make sure you're using the correct username
- Check that the user has SSH access (should be default)
- Try with `sudo` if needed (though not recommended for SSH)

## Quick Reference

```bash
# On Jetson - Start SSH
sudo systemctl start ssh
sudo systemctl enable ssh

# On Jetson - Find IP
hostname -I

# On Mac - Connect
ssh acre@<IP>

# On Mac - Copy files
scp -r ~/Documents/acre acre@<IP>:/home/acre/ACRE_Capstone/
```

