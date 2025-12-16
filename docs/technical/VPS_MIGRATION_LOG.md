# VPS Migration Log

## Migration Date: December 10, 2024

## Overview

Successfully migrated all data and services from old VPS (134.255.234.188) to new VPS (5.249.161.66) before the old server expired.

---

## Old VPS (Expired)

| Property | Value |
|----------|-------|
| **IP Address** | 134.255.234.188 |
| **OS** | Ubuntu 20.04 |
| **Node.js** | v20.x |
| **Python** | 3.8 |
| **Status** | EXPIRED / NO LONGER AVAILABLE |

---

## New VPS (Active)

| Property | Value |
|----------|-------|
| **IP Address** | 5.249.161.66 |
| **SSH Port** | 22 |
| **OS** | Debian 13 (Trixie) |
| **Kernel** | 6.12.41+deb13-amd64 |
| **Node.js** | v20.19.6 |
| **Python** | 3.13.5 |
| **RAM** | 18 GB |
| **Storage** | 493 GB (471 GB free after migration) |
| **Valid Until** | ~June 2025 (6 months) |

### Access Credentials

```bash
# Root access
ssh root@5.249.161.66
# Password: <VPS_PASSWORD>

# Developer user
ssh developer@5.249.161.66
# Password: "    " (4 spaces)
```

---

## Migration Process

### Step 1: SSH Key Setup
- Generated SSH key auth between old VPS and new VPS
- Added old VPS public key to new VPS authorized_keys

### Step 2: Create Developer User
```bash
useradd -m -s /bin/bash developer
echo "developer:    " | chpasswd
usermod -aG sudo developer
echo "developer ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
```

### Step 3: Install System Dependencies
```bash
# Node.js 20.x
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs

# Python 3.13 + pip
apt-get install -y python3 python3-pip python3-venv

# PM2 globally
npm install -g pm2
```

### Step 4: Data Transfer (rsync)
```bash
# From old VPS
rsync -avz --progress /home/developer/ root@5.249.161.66:/home/developer/
```

**Transfer Statistics:**
- Total data transferred: ~33 GB
- Transfer time: ~90 minutes
- Average speed: ~2-6 MB/s

### Step 5: Fix Ownership
```bash
chown -R developer:developer /home/developer/
```

### Step 6: Rebuild Frontend
```bash
cd /home/developer/3d-game-ai/frontend
npm install
npm run build
```

### Step 7: Recreate Backend Virtual Environment
```bash
cd /home/developer/3d-game-ai/backend
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install flask flask-cors flask-sqlalchemy gunicorn bcrypt pyjwt
```

### Step 8: Start PM2 Services
```bash
cd /home/developer/3d-game-ai
pm2 start ecosystem.config.js
pm2 save
pm2 startup
```

---

## Data Migrated

### Project Files (~33 GB total)

| Directory | Size | Description |
|-----------|------|-------------|
| `3d-game-ai/` | 3.1 GB | Frontend + Backend project |
| `voxformer_checkpoints/` | ~12 GB | Trained model checkpoints |
| `voxformer_backup/` | ~100 MB | VoxFormer source code |
| `best_final.pt` | 1.7 GB | Stage 1 final checkpoint |
| `malek/` | 7.7 GB | Other projects |
| `.cache/huggingface/` | ~3 GB | WavLM, E5 models |
| Other files | ~5 GB | Config, logs, misc |

### VoxFormer Checkpoints

| File | Size | Description |
|------|------|-------------|
| `best_final_stage1.pt` | 1.6 GB | Best Stage 1 checkpoint |
| `final_epoch19_step17840.pt` | 1.6 GB | Final epoch checkpoint |
| + 5 intermediate checkpoints | ~8 GB | Recovery options |

### VoxFormer Code Backup

```
voxformer_backup/
├── src/model/
│   ├── voxformer.py
│   ├── conformer.py
│   ├── zipformer.py
│   ├── decoder.py
│   └── wavlm_frontend.py
├── configs/
│   ├── stage1.yaml
│   ├── stage2.yaml
│   └── stage3_gaming.yaml
├── scripts/
├── tokenizer/
└── *.py (inference scripts)
```

---

## Services Running

### PM2 Process List

| ID | Name | Status | Port |
|----|------|--------|------|
| 0 | backend | online | 5000 |
| 1 | frontend | online | 3000 |

### ecosystem.config.js

```javascript
module.exports = {
  apps: [
    {
      name: 'backend',
      cwd: '/home/developer/3d-game-ai/backend',
      script: 'venv/bin/gunicorn',
      args: '-b 0.0.0.0:5000 app:app',
      interpreter: 'none',
      env: {
        VAULT_PASSWORD: 'admin123'
      }
    },
    {
      name: 'frontend',
      cwd: '/home/developer/3d-game-ai/frontend',
      script: 'npm',
      args: 'start',
      interpreter: 'none',
      env: {
        PORT: 3000
      }
    }
  ]
};
```

---

## Post-Migration Tasks

### Completed
- [x] SSH key setup
- [x] Developer user created
- [x] Node.js v20 installed
- [x] Python 3.13 installed
- [x] PM2 installed globally
- [x] All data transferred (33 GB)
- [x] File ownership fixed
- [x] Frontend rebuilt
- [x] Backend venv recreated
- [x] PM2 services running
- [x] PM2 auto-start on boot configured

### TODO (Optional)
- [ ] Update `.env.local` with new API URL if needed
- [ ] Update CORS settings in backend if accessing from different domains
- [ ] Set up SSL certificates (Let's Encrypt) for HTTPS
- [ ] Configure firewall (ufw)
- [ ] Set up automated backups

---

## Verification

### Frontend Test
```bash
curl -s http://5.249.161.66:3000 | head -5
# Returns HTML with "3D Game Generation AI Assistant"
```

### Backend Test
```bash
curl -s http://5.249.161.66:5000
# Returns 404 (expected - no root route defined)
# API endpoints like /api/tasks work correctly
```

---

## Troubleshooting

### Backend Crashes
If backend keeps erroring, check for missing Python packages:
```bash
pm2 logs backend --lines 50
# Look for "ModuleNotFoundError"
# Install missing package:
cd /home/developer/3d-game-ai/backend
source venv/bin/activate
pip install <missing-package>
pm2 restart backend
```

### Frontend Build Fails
```bash
cd /home/developer/3d-game-ai/frontend
rm -rf node_modules .next
npm install
npm run build
pm2 restart frontend
```

### Check Service Status
```bash
pm2 status
pm2 logs --lines 100
```

---

## Quick Reference Commands

```bash
# SSH to new VPS
ssh root@5.249.161.66  # Password: <VPS_PASSWORD>

# Check services
pm2 status

# View logs
pm2 logs frontend
pm2 logs backend

# Restart services
pm2 restart all

# Deploy frontend changes
cd /home/developer/3d-game-ai/frontend
npm run build
pm2 restart frontend
```

---

*Migration completed: December 10, 2024 18:30 UTC*
*Document created by Claude Code*
