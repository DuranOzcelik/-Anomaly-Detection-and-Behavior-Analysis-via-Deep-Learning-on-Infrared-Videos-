# Google Drive Service Account Setup Guide

## Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click on the project dropdown at the top
3. Click "NEW PROJECT"
4. Enter project name: `ThesisApp Thermal Detection`
5. Click "CREATE"
6. Wait for project to be created (this may take a minute)

## Step 2: Enable Google Drive API

1. In the Google Cloud Console, go to **APIs & Services** → **Library**
2. Search for `Google Drive API`
3. Click on it
4. Click the **ENABLE** button
5. Wait for it to be enabled

## Step 3: Create Service Account

1. Go to **APIs & Services** → **Credentials**
2. Click **+ CREATE CREDENTIALS** button
3. Select **Service Account**
4. Fill in the form:
   - Service account name: `thesis-app-drive-service`
   - Service account ID: (auto-generated)
   - Description: `Service account for ThesisApp to read videos from Google Drive`
5. Click **CREATE AND CONTINUE**
6. Skip the optional steps (you don't need to grant additional permissions now)
7. Click **DONE**

## Step 4: Create and Download JSON Key

1. In **APIs & Services** → **Credentials**
2. Under "Service Accounts", click on the service account you just created (`thesis-app-drive-service`)
3. Go to the **KEYS** tab
4. Click **+ ADD KEY** → **Create new key**
5. Select **JSON**
6. Click **CREATE**
7. A JSON file will download automatically. **Save it safely**

## Step 5: Configure Backend with Credentials

1. Copy the downloaded JSON file to your project:
   ```bash
   cp ~/Downloads/[downloaded-file].json ~/Desktop/ThesisApp/backend/google_drive_credentials.json
   ```

2. Create `.env` file in `backend/` directory:
   ```bash
   cd ~/Desktop/ThesisApp/backend
   echo "GOOGLE_DRIVE_CREDENTIALS_PATH=google_drive_credentials.json" > .env
   ```

3. Verify the file exists:
   ```bash
   ls -la google_drive_credentials.json
   # Output: -rw-r--r--  1 user  staff  2345 May 13 13:00 google_drive_credentials.json
   ```

## Step 6: Share Google Drive Folder with Service Account

1. Open the JSON file you downloaded and find the `client_email` field:
   ```json
   {
     "type": "service_account",
     "project_id": "...",
     "private_key_id": "...",
     "private_key": "...",
     "client_email": "thesis-app-drive-service@project-id.iam.gserviceaccount.com",  ← THIS ONE
     ...
   }
   ```

2. Go to Google Drive and navigate to `MyDrive/archive/LTD Dataset`
3. Right-click on the folder → **Share**
4. Paste the `client_email` from above into the share field
5. Give it **Editor** access (so it can read files)
6. Uncheck "Notify people" (it's a service account)
7. Click **Share**

## Step 7: Verify Connection

Run this test in the backend:

```bash
cd ~/Desktop/ThesisApp/backend
python3 -c "
from services.drive_service import get_drive_service
try:
    service = get_drive_service()
    print('✓ Google Drive connection successful!')
    videos = service.list_videos_by_date_range('20240101', '20240131')
    print(f'✓ Found {len(videos)} videos')
except Exception as e:
    print(f'✗ Error: {e}')
"
```

## Troubleshooting

### "GOOGLE_DRIVE_CREDENTIALS_PATH not set"
- Make sure `.env` file exists in `backend/` directory
- Run: `source .env` to load variables

### "File not found" error
- Verify the JSON file path is correct
- Check that the file is readable: `cat google_drive_credentials.json`

### "Permission denied" error
- Check that the Google Drive folder was shared with the `client_email`
- Verify the service account has at least **Viewer** access

### "Folder not found: archive"
- Verify the exact folder structure matches: `MyDrive/archive/LTD Dataset/LTD Dataset/Video Clips/YYYYMMDD/`
- Check folder names are exactly as shown (case-sensitive)

## Next Steps

After setup is complete:
1. Install Python dependencies: `pip install -r requirements.txt`
2. Start backend: `python3 -m uvicorn main:app --reload --host 0.0.0.0`
3. Backend will be available at: `http://localhost:8000`
4. WebSocket available at: `ws://localhost:8000/ws`
5. Run Flutter frontend and connect to backend

## Notes

- **Security**: Never commit `google_drive_credentials.json` to Git. Add it to `.gitignore`
- **Quota**: Service accounts have quota limits. Monitor usage in Google Cloud Console
- **Expiration**: Service account keys don't expire by default, but it's good practice to rotate them periodically
