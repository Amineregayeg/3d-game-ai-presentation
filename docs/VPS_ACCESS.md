# VPS Access & Security Configuration

> **Last Updated**: January 3, 2026
> **Provider**: ZAP Hosting
> **OS**: Ubuntu 20.04.6 LTS (Focal Fossa)

## Server Details

| Property | Value |
|----------|-------|
| **Host** | `5.249.161.66` |
| **SSH Port** | `22` |
| **Hostname** | `gold-raccoon-61739` |
| **RAM** | 31 GB |
| **Disk** | 492 GB |
| **OS** | Ubuntu 20.04.6 LTS |

## Access Methods

### 1. SSH Key Authentication (Preferred)

**Local SSH Key Location:**
```
Private Key: ~/.ssh/vps_5.249.161.66
Public Key:  ~/.ssh/vps_5.249.161.66.pub
```

**Quick Connect:**
```bash
# Using SSH config alias
ssh vps-zap

# Or directly with key
ssh -i ~/.ssh/vps_5.249.161.66 root@5.249.161.66
```

**SSH Config Entry** (`~/.ssh/config`):
```
Host vps-zap
    HostName 5.249.161.66
    User root
    Port 22
    IdentityFile ~/.ssh/vps_5.249.161.66
    StrictHostKeyChecking no
```

**Public Key (for adding to other servers):**
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKbKNPKxsnrv/Qq8DVblWWfGZBhUpijAp8b5iIBWz0rC claude-code-vps-access
```

### 2. Password Authentication (Fallback)

```bash
# Using sshpass
sshpass -p 'AiDev123123123.' ssh root@5.249.161.66

# Or manual
ssh root@5.249.161.66
# Password: AiDev123123123.
```

## Firewall Configuration (UFW)

**Status**: Active and enabled on boot

| Port | Protocol | Service |
|------|----------|---------|
| 22 | TCP | SSH |
| 80 | TCP | HTTP |
| 443 | TCP | HTTPS |
| 3000 | TCP | Next.js Frontend |
| 5000 | TCP | Flask Backend |
| 5001 | TCP | GPU Tunnel - Whisper/SadTalker |
| 5002 | TCP | GPU Tunnel - VoxFormer |
| 9876 | TCP | Blender MCP Server |

**Manage Firewall:**
```bash
# Check status
ufw status verbose

# Add new port
ufw allow <port>/tcp comment 'Description'

# Remove port
ufw delete allow <port>/tcp

# Disable (emergency only)
ufw disable
```

## Security Tools Installed

### 1. rkhunter (Rootkit Hunter)
```bash
# Run scan
rkhunter --check --skip-keypress

# Update database
rkhunter --update
rkhunter --propupd
```

### 2. chkrootkit
```bash
# Quick scan
chkrootkit -q

# Full scan
chkrootkit
```

### 3. ClamAV (Antivirus)
```bash
# Scan a directory
clamscan -r /path/to/scan

# Update virus definitions
freshclam
```

## SSH Hardening Applied

**Configuration**: `/etc/ssh/sshd_config.d/hardening.conf`

| Setting | Value | Notes |
|---------|-------|-------|
| PasswordAuthentication | yes | Fallback enabled |
| PubkeyAuthentication | yes | Primary method |
| PermitRootLogin | yes | Direct root access |
| MaxAuthTries | 10 | Generous limit |
| MaxSessions | 20 | Multiple sessions allowed |
| LoginGraceTime | 120s | 2 minutes to login |
| ClientAliveInterval | 300s | Keep connections alive |

**No aggressive banning** - designed to prevent lockouts.

## Quick Commands Reference

```bash
# Connect to VPS
ssh vps-zap

# Check system status
ssh vps-zap "uptime && df -h / && free -h"

# View running services
ssh vps-zap "pm2 status"

# Run security scan
ssh vps-zap "chkrootkit -q"

# Check firewall
ssh vps-zap "ufw status"

# View recent logins
ssh vps-zap "last -10"

# Check for failed SSH attempts
ssh vps-zap "grep 'Failed password' /var/log/auth.log | tail -20"
```

## Scheduled Maintenance

Consider setting up:
```bash
# Weekly rkhunter scan (crontab -e)
0 3 * * 0 /usr/bin/rkhunter --check --skip-keypress --report-warnings-only

# Daily ClamAV update
0 4 * * * /usr/bin/freshclam --quiet
```

## Recovery Procedures

### If Locked Out via SSH:
1. Access ZAP Hosting Dashboard
2. Use VNC Console
3. Login with root credentials
4. Check `/var/log/auth.log` for issues
5. Verify SSH service: `systemctl status ssh`
6. Check firewall: `ufw status`

### If Firewall Blocks Access:
```bash
# Via VNC console
ufw allow 22/tcp
ufw reload
```

### Reset SSH to Defaults:
```bash
# Via VNC console
rm /etc/ssh/sshd_config.d/hardening.conf
systemctl restart ssh
```

## Credential Summary

| Access Type | Credentials |
|-------------|-------------|
| SSH (Key) | `~/.ssh/vps_5.249.161.66` |
| SSH (Password) | `root` / `AiDev123123123.` |
| ZAP Dashboard | (user's account) |

---

**Security Note**: This file contains sensitive credentials. Keep it secure and do not commit to public repositories without redacting passwords.
