#!/usr/bin/env bash
# ==============================================================================
# ISRO Lunar Image Registration Pipeline — Native macOS ARM64 DMG Builder
# ==============================================================================
# Packages the FastAPI backend and Streamlit frontend into a native macOS
# application bundle (.app) and generates a distributable Apple Silicon ARM64
# .dmg installer using create-dmg.
#
# Usage:
#   ./scripts/build_dmg.sh [OPTIONS]
#
# Options:
#   --clean             Clean previous build artifacts before building
#   --rebuild-backend   Force rebuild of the compiled PyInstaller backend binary
#   --include-runtime   Bundle full Python virtual environment payload into .app
#   --skip-dmg          Build the .app bundle only, skip create-dmg step
#   --help              Display this help message
# ==============================================================================

set -euo pipefail

# ── Color Output Utilities ───────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

log_info()    { echo -e "${BLUE}${BOLD}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}${BOLD}[SUCCESS]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}${BOLD}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}${BOLD}[ERROR]${NC} $1" >&2; }
log_step()    { echo -e "\n${CYAN}${BOLD}==> $1${NC}"; }

# ── Paths & Variables ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

APP_NAME="ISRO Lunar Registration"
BUNDLE_ID="gov.isro.sac.lunar-registration"
VERSION="1.0.0"
DIST_DIR="$PROJECT_ROOT/dist"
BUILD_DIR="$PROJECT_ROOT/build"
APP_BUNDLE="$DIST_DIR/$APP_NAME.app"
STAGING_DIR="$DIST_DIR/dmg_staging"
DMG_OUTPUT="$DIST_DIR/ISRO-Lunar-Registration-ARM64.dmg"
VENV_PYTHON="$PROJECT_ROOT/venv/bin/python"

# CLI Flags
DO_CLEAN=false
REBUILD_BACKEND=false
INCLUDE_RUNTIME=false
SKIP_DMG=false

for arg in "$@"; do
    case "$arg" in
        --clean)
            DO_CLEAN=true
            ;;
        --rebuild-backend)
            REBUILD_BACKEND=true
            ;;
        --include-runtime)
            INCLUDE_RUNTIME=true
            ;;
        --skip-dmg)
            SKIP_DMG=true
            ;;
        --help|-h)
            echo "Usage: ./scripts/build_dmg.sh [--clean] [--rebuild-backend] [--include-runtime] [--skip-dmg]"
            exit 0
            ;;
        *)
            log_error "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

# ── Pre-flight Checks ────────────────────────────────────────────────────────
log_step "Step 1: Running Pre-flight Environment Checks & Workspace Clean"

# 1. Host OS check
OS_TYPE="$(uname -s)"
if [ "$OS_TYPE" != "Darwin" ]; then
    log_error "This script must be run on macOS (Darwin). Detected: $OS_TYPE"
    exit 1
fi

# 2. Apple Silicon ARM64 check
ARCH_TYPE="$(uname -m)"
if [ "$ARCH_TYPE" != "arm64" ]; then
    log_warn "Current architecture is $ARCH_TYPE, expected arm64 (Apple Silicon). Proceeding with build..."
else
    log_info "Verified Apple Silicon architecture: $ARCH_TYPE"
fi

# 3. Virtualenv check
if [ ! -f "$VENV_PYTHON" ]; then
    log_error "Python virtual environment not found at $VENV_PYTHON."
    log_error "Please create it using: python3 -m venv venv && ./venv/bin/pip install -r requirements.txt"
    exit 1
fi
log_info "Using Python interpreter: $("$VENV_PYTHON" --version)"

# 4. create-dmg check
CREATE_DMG_BIN=""
if command -v create-dmg >/dev/null 2>&1; then
    CREATE_DMG_BIN="$(command -v create-dmg)"
elif [ -f "/opt/homebrew/bin/create-dmg" ]; then
    CREATE_DMG_BIN="/opt/homebrew/bin/create-dmg"
elif [ -f "/usr/local/bin/create-dmg" ]; then
    CREATE_DMG_BIN="/usr/local/bin/create-dmg"
fi

if [ "$SKIP_DMG" = false ]; then
    if [ -z "$CREATE_DMG_BIN" ]; then
        log_error "create-dmg not found in PATH or Homebrew locations."
        log_error "Install it via: brew install create-dmg"
        exit 1
    fi
    log_info "Found create-dmg utility: $CREATE_DMG_BIN ($("$CREATE_DMG_BIN" --version 2>/dev/null || echo 'installed'))"
fi

# 5. Clean junk files across workspace
log_info "Purging junk cache files, temp mounts, and stale build artifacts..."
rm -rf "$BUILD_DIR" "$STAGING_DIR"
rm -f "$DIST_DIR"/rw.*.dmg "$PROJECT_ROOT"/*.spec 2>/dev/null || true
find "$PROJECT_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$PROJECT_ROOT" -type f -name "*.pyc" -delete 2>/dev/null || true
find "$PROJECT_ROOT" -type f -name ".DS_Store" -delete 2>/dev/null || true

# Force detach any lingering DMG mounts from previous interrupted builds
for dev in $(hdiutil info | grep -B 1 "ISRO Lunar Registration" | grep "/dev/disk" | awk '{print $1}'); do
    log_info "Detaching stale disk image: $dev"
    hdiutil detach "$dev" -force >/dev/null 2>&1 || true
done

if [ "$DO_CLEAN" = true ]; then
    log_info "Performing full clean (--clean passed)..."
    rm -rf "$APP_BUNDLE" "$DIST_DIR/isro-backend" "$DMG_OUTPUT"
    log_success "Clean completed."
fi

mkdir -p "$DIST_DIR"

# ── Asset Generation ─────────────────────────────────────────────────────────
log_step "Step 2: Generating Native macOS Branding Assets (.icns & DMG layout)"
ASSETS_DIR="$PROJECT_ROOT/assets"
mkdir -p "$ASSETS_DIR"

if [ ! -f "$ASSETS_DIR/AppIcon.icns" ] || [ ! -f "$ASSETS_DIR/dmg_background.png" ]; then
    log_info "Generating AppIcon.icns and dmg_background.png via Pillow & iconutil..."
    "$VENV_PYTHON" "$SCRIPT_DIR/generate_assets.py"
    log_success "Visual assets generated successfully."
else
    log_info "Existing visual assets verified in assets/ directory."
fi

# ── Package the Backend ──────────────────────────────────────────────────────
log_step "Step 3: Packaging Backend REST Service (FastAPI + PyTorch/Vision/Kornia)"

BACKEND_DIST_DIR="$DIST_DIR/isro-backend"
BACKEND_EXEC="$BACKEND_DIST_DIR/isro-backend"

if [ "$REBUILD_BACKEND" = true ] || [ ! -f "$BACKEND_EXEC" ]; then
    log_info "Building standalone ARM64 backend binary with PyInstaller..."
    "$VENV_PYTHON" -m PyInstaller \
        --onedir \
        --name isro-backend \
        --paths "$PROJECT_ROOT" \
        --add-data "configs:configs" \
        --collect-all api \
        --collect-all src \
        "$PROJECT_ROOT/run_backend.py" \
        --noconfirm
    log_success "Backend executable built at: $BACKEND_EXEC"
else
    log_info "Existing compiled backend found at: $BACKEND_EXEC (use --rebuild-backend to refresh)"
fi

# ── Assemble macOS Application Bundle (.app) ─────────────────────────────────
log_step "Step 4: Assembling Native macOS Application Bundle ($APP_NAME.app)"

CONTENTS_DIR="$APP_BUNDLE/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"
APP_PAYLOAD_DIR="$RESOURCES_DIR/app"
BIN_DIR="$RESOURCES_DIR/bin"

rm -rf "$APP_BUNDLE"
mkdir -p "$MACOS_DIR" "$RESOURCES_DIR" "$APP_PAYLOAD_DIR" "$BIN_DIR"

# Copy Icon
cp "$ASSETS_DIR/AppIcon.icns" "$RESOURCES_DIR/AppIcon.icns"

# Copy Compiled Backend Binary
log_info "Installing compiled backend into app bundle..."
cp -R "$BACKEND_DIST_DIR/"* "$BIN_DIR/"

# Copy Frontend and Application Source Code
log_info "Installing Streamlit frontend, API modules, and configs..."
cp -R "$PROJECT_ROOT/ui" "$APP_PAYLOAD_DIR/"
cp -R "$PROJECT_ROOT/api" "$APP_PAYLOAD_DIR/"
cp -R "$PROJECT_ROOT/src" "$APP_PAYLOAD_DIR/"
cp -R "$PROJECT_ROOT/configs" "$APP_PAYLOAD_DIR/"
cp "$PROJECT_ROOT/run_backend.py" "$APP_PAYLOAD_DIR/"

# Copy curated browse images and metadata (excluding 20+ GB raw sensor arrays)
if [ -d "$PROJECT_ROOT/data" ]; then
    log_info "Installing curated dataset previews and metadata..."
    mkdir -p "$APP_PAYLOAD_DIR/data"
    cp -R "$PROJECT_ROOT/data/labels" "$APP_PAYLOAD_DIR/data/" 2>/dev/null || true
    cp -R "$PROJECT_ROOT/data/calibrated" "$APP_PAYLOAD_DIR/data/" 2>/dev/null || true
    
    mkdir -p "$APP_PAYLOAD_DIR/data/raw"
    find "$PROJECT_ROOT/data/raw" -type f -name "*_brw_*.png" | while read -r img; do
        rel="${img#"$PROJECT_ROOT/data/raw/"}"
        dest_dir="$APP_PAYLOAD_DIR/data/raw/$(dirname "$rel")"
        mkdir -p "$dest_dir"
        cp "$img" "$dest_dir/"
    done
fi

# Streamlit Production Configuration
mkdir -p "$APP_PAYLOAD_DIR/.streamlit"
cat << 'EOF' > "$APP_PAYLOAD_DIR/.streamlit/config.toml"
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false
runOnSave = false
maxUploadSize = 1024

[browser]
gatherUsageStats = false
serverAddress = "127.0.0.1"

[theme]
base = "dark"
primaryColor = "#6C63FF"
backgroundColor = "#0F172A"
secondaryBackgroundColor = "#1E293B"
textColor = "#F8FAFC"
font = "sans serif"
EOF

# Optional: Bundle Python Runtime
if [ "$INCLUDE_RUNTIME" = true ]; then
    log_info "Copying Python virtualenv into app bundle runtime (standalone mode)..."
    RUNTIME_DIR="$RESOURCES_DIR/runtime"
    mkdir -p "$RUNTIME_DIR"
    cp -R "$PROJECT_ROOT/venv/lib" "$RUNTIME_DIR/"
    cp -R "$PROJECT_ROOT/venv/include" "$RUNTIME_DIR/" 2>/dev/null || true
    mkdir -p "$RUNTIME_DIR/bin"
    cp "$PROJECT_ROOT/venv/bin/python"* "$RUNTIME_DIR/bin/" 2>/dev/null || true
    cp "$PROJECT_ROOT/venv/pyvenv.cfg" "$RUNTIME_DIR/"
fi

# ── Link Backend and Frontend in Native Launcher ─────────────────────────────
log_step "Step 5: Creating Native Process Supervisor & Lifecycle Launcher"

LAUNCHER_SCRIPT="$MACOS_DIR/$APP_NAME"
cat << 'EOF' > "$LAUNCHER_SCRIPT"
#!/bin/bash
# ==============================================================================
# ISRO Lunar Registration — Process Supervisor & Native Launcher
# ==============================================================================
set -e

APP_BUNDLE="$(cd "$(dirname "$0")/../.." && pwd)"
CONTENTS="$APP_BUNDLE/Contents"
RESOURCES="$CONTENTS/Resources"
APP_DIR="$RESOURCES/app"
BIN_DIR="$RESOURCES/bin"
LOG_DIR="$HOME/Library/Logs/ISRO-Lunar-Registration"
mkdir -p "$LOG_DIR"

BACKEND_PORT=8000
FRONTEND_PORT=8501
export BACKEND_API_URL="http://127.0.0.1:${BACKEND_PORT}"
export OPENCV_IO_MAX_IMAGE_PIXELS="1099511627776"

# Detect StereoPipeline if present
if [ -d "/Users/mahim/Desktop/moon project ISRO copy/StereoPipeline-3.8.0-alpha-2026-08-31-arm64-OSX/bin" ]; then
    export PATH="/Users/mahim/Desktop/moon project ISRO copy/StereoPipeline-3.8.0-alpha-2026-08-31-arm64-OSX/bin:$PATH"
fi

# Cleanup function for graceful termination
BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    echo "[*] Terminating ISRO Lunar Registration..."
    if [ -n "$FRONTEND_PID" ]; then
        kill -TERM "$FRONTEND_PID" 2>/dev/null || true
    fi
    if [ -n "$BACKEND_PID" ]; then
        kill -TERM "$BACKEND_PID" 2>/dev/null || true
    fi
    # Free ports if still bound
    lsof -ti:${FRONTEND_PORT} 2>/dev/null | xargs kill -9 2>/dev/null || true
    lsof -ti:${BACKEND_PORT} 2>/dev/null | xargs kill -9 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM SIGHUP EXIT

# Clean up any stale instances on target ports
lsof -ti:${BACKEND_PORT} 2>/dev/null | xargs kill -9 2>/dev/null || true
lsof -ti:${FRONTEND_PORT} 2>/dev/null | xargs kill -9 2>/dev/null || true

# Determine Python runtime
if [ -f "$RESOURCES/runtime/bin/python3" ]; then
    PYTHON_BIN="$RESOURCES/runtime/bin/python3"
elif [ -f "/Users/mahim/Desktop/moon project ISRO copy/venv/bin/python" ]; then
    PYTHON_BIN="/Users/mahim/Desktop/moon project ISRO copy/venv/bin/python"
else
    PYTHON_BIN="$(which python3)"
fi

# 1. Launch Backend REST Service
echo "[1/3] Starting FastAPI Backend on http://127.0.0.1:${BACKEND_PORT}..."
if [ -f "$BIN_DIR/isro-backend" ]; then
    "$BIN_DIR/isro-backend" --host 127.0.0.1 --port "$BACKEND_PORT" > "$LOG_DIR/backend.log" 2>&1 &
    BACKEND_PID=$!
else
    PYTHONPATH="$APP_DIR" "$PYTHON_BIN" "$APP_DIR/run_backend.py" --host 127.0.0.1 --port "$BACKEND_PORT" > "$LOG_DIR/backend.log" 2>&1 &
    BACKEND_PID=$!
fi

# 2. Wait for Backend Health Check
HEALTH_URL="http://127.0.0.1:${BACKEND_PORT}/api/v1/health"
MAX_RETRIES=40
RETRY_COUNT=0
BACKEND_READY=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s -f "$HEALTH_URL" >/dev/null 2>&1; then
        BACKEND_READY=true
        break
    fi
    sleep 0.5
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ "$BACKEND_READY" = true ]; then
    echo "[✓] Backend REST API is online!"
else
    echo "[!] Backend health check timed out, proceeding to launch UI in fallback mode..."
fi

# 3. Launch Streamlit Frontend
echo "[2/3] Starting Streamlit Frontend on http://127.0.0.1:${FRONTEND_PORT}..."
cd "$APP_DIR"
PYTHONPATH="$APP_DIR:${PYTHONPATH:-}" "$PYTHON_BIN" -m streamlit run \
    "$APP_DIR/ui/app.py" \
    --server.port "$FRONTEND_PORT" \
    --server.address 127.0.0.1 \
    --server.headless true \
    --browser.gatherUsageStats false \
    > "$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!

# 4. Open Default Web Browser
echo "[3/3] Opening user interface in default browser..."
sleep 1.5
open "http://127.0.0.1:${FRONTEND_PORT}"

# Native macOS Notification
osascript -e 'display notification "ISRO Lunar Image Registration Pipeline is ready at http://localhost:8501" with title "ISRO Lunar Registration" sound name "Glass"' 2>/dev/null || true

# Keep launcher alive while frontend is running
wait "$FRONTEND_PID"
EOF

chmod +x "$LAUNCHER_SCRIPT"
log_success "Native process launcher script created."

# ── Generate Info.plist & PkgInfo ─────────────────────────────────────────────
log_step "Step 6: Writing macOS Bundle Metadata (Info.plist & PkgInfo)"

cat << EOF > "$CONTENTS_DIR/Info.plist"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleDisplayName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleIdentifier</key>
    <string>${BUNDLE_ID}</string>
    <key>CFBundleVersion</key>
    <string>${VERSION}</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>ISRO</string>
    <key>CFBundleExecutable</key>
    <string>${APP_NAME}</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon.icns</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>LSArchitecturePriority</key>
    <array>
        <string>arm64</string>
    </array>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSSupportsAutomaticGraphicsSwitching</key>
    <true/>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.education</string>
    <key>NSHumanReadableCopyright</key>
    <string>© 2026 Space Applications Centre (SAC), ISRO. All rights reserved.</string>
</dict>
</plist>
EOF

echo -n "APPLISRO" > "$CONTENTS_DIR/PkgInfo"

# ── Code Signing ─────────────────────────────────────────────────────────────
log_step "Step 7: Applying macOS Ad-Hoc Code Signature"
log_info "Signing bundle for Apple Silicon ARM64 runtime authorization..."
codesign --force --deep --sign - "$APP_BUNDLE"
log_success "Code signing complete. Verification:"
codesign --verify --deep --strict "$APP_BUNDLE"

# ── DMG Generation via create-dmg ────────────────────────────────────────────
if [ "$SKIP_DMG" = false ]; then
    log_step "Step 8: Generating Native macOS ARM64 DMG using create-dmg"

    rm -rf "$STAGING_DIR"
    mkdir -p "$STAGING_DIR"
    log_info "Staging $APP_NAME.app for create-dmg..."
    cp -R "$APP_BUNDLE" "$STAGING_DIR/"

    rm -f "$DMG_OUTPUT"

    log_info "Executing create-dmg with high-res styling, window positioning, and Applications link..."
    
    # Run create-dmg pointing to the staging directory containing the .app bundle
    "$CREATE_DMG_BIN" \
        --volname "$APP_NAME" \
        --volicon "$ASSETS_DIR/AppIcon.icns" \
        --background "$ASSETS_DIR/dmg_background.png" \
        --window-pos 200 120 \
        --window-size 800 450 \
        --icon-size 110 \
        --icon "$APP_NAME.app" 200 220 \
        --hide-extension "$APP_NAME.app" \
        --app-drop-link 600 220 \
        --hdiutil-retries 15 \
        --overwrite \
        --no-internet-enable \
        "$DMG_OUTPUT" \
        "$STAGING_DIR"

    # Clean staging directory
    rm -rf "$STAGING_DIR"

    log_success "DMG successfully created at: $DMG_OUTPUT"

    # ── DMG Mount Verification ───────────────────────────────────────────────
    log_step "Step 9: Verifying Disk Image Integrity"
    MOUNT_DIR=$(mktemp -d /tmp/isro_dmg_mount.XXXXXX)
    hdiutil attach -nobrowse -readonly "$DMG_OUTPUT" -mountpoint "$MOUNT_DIR" >/dev/null
    
    if [ -d "$MOUNT_DIR/$APP_NAME.app" ] && [ -L "$MOUNT_DIR/Applications" ]; then
        log_success "Disk image verification PASSED! App bundle and Applications symlink are present."
    else
        log_error "Disk image verification FAILED! Content missing inside mounted volume."
        hdiutil detach "$MOUNT_DIR" -force >/dev/null 2>&1 || true
        exit 1
    fi
    
    hdiutil detach "$MOUNT_DIR" -force >/dev/null 2>&1
    rmdir "$MOUNT_DIR" 2>/dev/null || true
fi

# ── Summary Report ───────────────────────────────────────────────────────────
log_step "Build Complete!"
echo -e "${GREEN}${BOLD}========================================================================${NC}"
echo -e "${GREEN}${BOLD} 🎉 ISRO Lunar Registration macOS ARM64 Installer Built Successfully!${NC}"
echo -e "${GREEN}${BOLD}========================================================================${NC}"
echo -e "📦 ${BOLD}Application Bundle:${NC} $APP_BUNDLE"
if [ "$SKIP_DMG" = false ]; then
    DMG_SIZE="$(du -sh "$DMG_OUTPUT" | cut -f1)"
    DMG_SHA="$(shasum -a 256 "$DMG_OUTPUT" | cut -d' ' -f1)"
    echo -e "💿 ${BOLD}Final DMG Installer:${NC} $DMG_OUTPUT"
    echo -e "📏 ${BOLD}Installer Size:${NC}     $DMG_SIZE"
    echo -e "🔒 ${BOLD}SHA-256 Checksum:${NC}   $DMG_SHA"
fi
echo -e "\n${CYAN}${BOLD}To test the installed app directly:${NC}"
echo -e "  open \"$APP_BUNDLE\""
if [ "$SKIP_DMG" = false ]; then
    echo -e "\n${CYAN}${BOLD}To test the DMG installer:${NC}"
    echo -e "  open \"$DMG_OUTPUT\""
fi
echo ""
