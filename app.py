import os
import cv2
import json
import uuid
import time
import ailia
import psutil
import logging
import threading
import numpy as np
import subprocess
import urllib.request
import urllib.parse
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ===================================================================
# KONFIGURASI APLIKASI
# ===================================================================

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Inisialisasi Flask App
# Folder static dan template tetap, untuk file inti seperti CSS/JS jika ada
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Izinkan Cross-Origin Resource Sharing

# ===================================================================
# PERUBAHAN UNTUK RENDER: Konfigurasi Path Dinamis
# ===================================================================

# Cek apakah aplikasi berjalan di lingkungan Render dengan memeriksa variabel lingkungan
IS_ON_RENDER = 'RENDER' in os.environ

# Direktori dasar proyek
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if IS_ON_RENDER:
    # Gunakan Disk Persisten di Render untuk semua data yang perlu disimpan
    # Path ini harus cocok dengan Mount Path yang diatur di dashboard Render
    logger.info("Aplikasi berjalan di lingkungan Render, menggunakan disk persisten.")
    DATA_DIR = '/var/data/video_enhancer_data'
    MODEL_DIR = os.path.join(DATA_DIR, 'model')
    UPLOAD_FOLDER = os.path.join(DATA_DIR, 'uploads')
    ENHANCED_FOLDER = os.path.join(DATA_DIR, 'enhanced')
    HISTORY_FILE = os.path.join(DATA_DIR, 'history.json')
else:
    # Gunakan folder lokal jika tidak berjalan di Render (untuk development)
    logger.info("Aplikasi berjalan di lingkungan lokal.")
    MODEL_DIR = os.path.join(BASE_DIR, 'model')
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    ENHANCED_FOLDER = os.path.join(BASE_DIR, 'static', 'enhanced')
    HISTORY_FILE = os.path.join(BASE_DIR, 'history.json')

# Pastikan semua folder yang dibutuhkan ada, termasuk di disk persisten
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ENHANCED_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ===================================================================

# Variabel Global untuk state management
SESSIONS = {} # Menyimpan progres dari setiap sesi pemrosesan
SCALE = '4' # Skala super-resolution
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/edsr/'

# ===================================================================
# FUNGSI UTILITAS
# ===================================================================

def download_file(url, filepath, desc="Downloading"):
    """Download file dengan logging progres."""
    try:
        logger.info(f"📥 Mendownload: {os.path.basename(filepath)} dari {url}")
        urllib.request.urlretrieve(url, filepath)
        logger.info(f"✅ Download selesai: {os.path.basename(filepath)}")
        return True
    except Exception as e:
        logger.error(f"❌ Gagal download {os.path.basename(filepath)}: {e}")
        return False

def check_and_download_model(scale):
    """Cek keberadaan model dan download jika perlu."""
    weight_path = os.path.join(MODEL_DIR, f'edsr_scale{scale}.onnx')
    proto_path = os.path.join(MODEL_DIR, f'edsr_scale{scale}.onnx.prototxt')
    
    if not os.path.exists(weight_path) or not os.path.exists(proto_path):
        logger.info(f"🔄 Model scale {scale} tidak ditemukan, memulai download...")
        weight_url = urllib.parse.urljoin(REMOTE_PATH, f'edsr_scale{scale}.onnx')
        proto_url = urllib.parse.urljoin(REMOTE_PATH, f'edsr_scale{scale}.onnx.prototxt')
        
        if not download_file(weight_url, weight_path, f"Weight file scale {scale}"): return False
        if not download_file(proto_url, proto_path, f"Proto file scale {scale}"): return False
    
    logger.info(f"✅ Model scale {scale} sudah tersedia.")
    return True

def merge_audio(temp_video, original_video, output_video):
    """Menggabungkan audio dari video asli ke video hasil proses."""
    try:
        command = [
            "ffmpeg", "-y",
            "-i", temp_video,
            "-i", original_video,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            output_video
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ FFmpeg error: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("❌ FFmpeg tidak ditemukan. Pastikan FFmpeg terinstall di sistem Anda.")
        return False

def save_history(data):
    """Menyimpan data riwayat ke file JSON di lokasi yang benar."""
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    history.insert(0, data)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

def sharpen_image(image):
    """Menerapkan filter penajaman sederhana."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)
    
# ===================================================================
# LOGIKA INTI PEMROSESAN VIDEO
# ===================================================================

def get_url_for_file(folder_name, filename):
    """Membuat URL yang benar tergantung lingkungan (Render atau lokal)."""
    if IS_ON_RENDER:
        # Di Render, file disajikan melalui endpoint '/data/...'
        return f"/data/{folder_name}/{filename}"
    else:
        # Di lokal, file disajikan dari folder '/static/...'
        return f"/static/{folder_name}/{filename}"

def process_video_task(session_id, original_video_path, original_filename):
    """
    Fungsi ini berjalan di background thread untuk memproses video.
    """
    global SESSIONS
    
    session_data = SESSIONS[session_id]
    start_time = time.time()
    
    try:
        # Inisialisasi model, buka video, dll. (tidak ada perubahan di sini)
        session_data['status'] = "Initializing AI model..."
        if not check_and_download_model(SCALE):
            raise RuntimeError("Gagal mengunduh model AI.")
        
        WEIGHT_PATH = os.path.join(MODEL_DIR, f'edsr_scale{SCALE}.onnx')
        MODEL_PATH = os.path.join(MODEL_DIR, f'edsr_scale{SCALE}.onnx.prototxt')
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=-1)
        
        capture = cv2.VideoCapture(original_video_path)
        if not capture.isOpened():
            raise RuntimeError("Gagal membuka file video.")
        
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        scale_factor = int(SCALE)
        new_w, new_h = w * scale_factor, h * scale_factor
        
        unique_id = os.path.basename(original_video_path).split('.')[0]
        temp_video_filename = f"{unique_id}_temp.mp4"
        final_video_filename = f"{unique_id}_enhanced.mp4"
        
        temp_video_path = os.path.join(ENHANCED_FOLDER, temp_video_filename)
        final_video_path = os.path.join(ENHANCED_FOLDER, final_video_filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (new_w, new_h))
        
        frame_idx = 0
        psnr_scores, ssim_scores = [], []
        
        while True:
            ret, frame = capture.read()
            if not ret: break
            
            frame_idx += 1
            session_data['progress'] = int((frame_idx / total_frames) * 90)
            session_data['status'] = f"Processing frame {frame_idx}/{total_frames}"

            input_data = frame.astype(np.float32).transpose(2, 0, 1)
            input_data = np.expand_dims(input_data, axis=0)
            net.set_input_shape((1, 3, h, w))
            preds_ailia = net.predict(input_data)[0]
            
            output_img = preds_ailia.transpose(1, 2, 0)
            output_img = np.clip(output_img, 0, 255).astype(np.uint8)
            output_img = sharpen_image(output_img)
            writer.write(output_img)
            
            original_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            psnr_val = psnr(original_resized, output_img, data_range=255)
            ssim_val = ssim(original_resized, output_img, data_range=255, multichannel=True, channel_axis=2)
            psnr_scores.append(psnr_val)
            ssim_scores.append(ssim_val)
            
            session_data['current_psnr'] = psnr_val
            session_data['current_ssim'] = ssim_val

        capture.release()
        writer.release()
        
        session_data['status'] = "Merging audio..."
        session_data['progress'] = 95
        merge_audio(temp_video_path, original_video_path, final_video_path)
        
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

        end_time = time.time()
        processing_time = round(end_time - start_time, 2)
        metrics = {'psnr_avg': f"{np.mean(psnr_scores):.2f}", 'ssim_avg': f"{np.mean(ssim_scores):.4f}", 'psnr_per_frame': psnr_scores, 'ssim_per_frame': ssim_scores}
        
        final_result = {
            # PERUBAHAN: Gunakan fungsi helper untuk mendapatkan URL yang benar
            'enhanced_video_url': get_url_for_file('enhanced', final_video_filename),
            'processing_time': str(processing_time), 'metrics': metrics
        }
        session_data['final_result'] = final_result
        session_data['progress'] = 100
        session_data['status'] = "Processing complete!"

        history_data = {
            'id': str(uuid.uuid4()),
            'original_filename': original_filename,
            # PERUBAHAN: Gunakan fungsi helper untuk mendapatkan URL yang benar
            'original_video_url': get_url_for_file('uploads', os.path.basename(original_video_path)),
            'enhanced_video_url': get_url_for_file('enhanced', final_video_filename),
            'processing_time': str(processing_time),
            'metrics': {'psnr_avg': metrics['psnr_avg'], 'ssim_avg': metrics['ssim_avg']},
            'timestamp': time.time()
        }
        save_history(history_data)

    except Exception as e:
        logger.error(f"Error in session {session_id}: {e}", exc_info=True)
        SESSIONS[session_id]['error'] = str(e)
        SESSIONS[session_id]['status'] = "An error occurred."

# ===================================================================
# ENDPOINTS / ROUTES
# ===================================================================

@app.route('/')
def index():
    """Menyajikan halaman utama (frontend)."""
    return send_from_directory(app.template_folder, 'index.html')

# ===================================================================
# PENAMBAHAN UNTUK RENDER: Endpoint untuk menyajikan file dari Disk Persisten
# ===================================================================
@app.route('/data/<folder>/<filename>')
def serve_persistent_file(folder, filename):
    """Menyajikan file dari folder data persisten (uploads/enhanced)."""
    # Pastikan path aman dan sesuai
    if folder not in ['uploads', 'enhanced']:
        return "Not Found", 404
    
    # Pilih direktori yang benar berdasarkan parameter 'folder'
    directory = UPLOAD_FOLDER if folder == 'uploads' else ENHANCED_FOLDER
    
    return send_from_directory(directory, filename)
# ===================================================================


@app.route('/process', methods=['POST'])
def process_endpoint():
    """Menerima video dan memulai pemrosesan."""
    if 'video' not in request.files: return jsonify({'error': 'No video file provided'}), 400
    file = request.files['video']
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400
    
    ext = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{ext}"
    video_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(video_path)

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {'progress': 0, 'status': 'Waiting to start...', 'error': None, 'final_result': None, 'current_psnr': 0, 'current_ssim': 0}
    
    thread = threading.Thread(target=process_video_task, args=(session_id, video_path, file.filename))
    thread.start()
    
    return jsonify({'session_id': session_id})

@app.route('/progress/<session_id>')
def progress_stream(session_id):
    """Mengirim progres pemrosesan menggunakan Server-Sent Events (SSE)."""
    def generate():
        while session_id in SESSIONS and SESSIONS[session_id].get('error') is None:
            data = SESSIONS[session_id]
            yield f"data: {json.dumps(data)}\n\n"
            if data.get('progress', 0) >= 100: break
            time.sleep(1)
        
        if session_id in SESSIONS:
             yield f"data: {json.dumps(SESSIONS[session_id])}\n\n"
             time.sleep(5)
             del SESSIONS[session_id]

    return Response(generate(), mimetype='text/event-stream')

@app.route('/history')
def get_history():
    """Mengambil data riwayat pemrosesan dari file JSON."""
    if not os.path.exists(HISTORY_FILE): return jsonify([])
    try:
        with open(HISTORY_FILE, 'r') as f:
            return jsonify(json.load(f))
    except (json.JSONDecodeError, FileNotFoundError):
        return jsonify([])

@app.route('/history/delete', methods=['POST'])
def delete_history_item():
    """Menghapus item riwayat dan file video terkait."""
    data = request.get_json()
    identifier = data.get('identifier')
    if not identifier: return jsonify({'error': 'Identifier diperlukan'}), 400
    if not os.path.exists(HISTORY_FILE): return jsonify({'message': 'File riwayat tidak ditemukan.'}), 200

    with open(HISTORY_FILE, 'r') as f:
        try: history = json.load(f)
        except json.JSONDecodeError: history = []

    item_to_delete = None
    index_to_delete = -1
    for i, item in enumerate(history):
        if str(item.get('id')) == str(identifier) or item.get('original_video_url') == identifier:
            item_to_delete = item
            index_to_delete = i
            break
            
    if not item_to_delete:
        logger.warning(f"Percobaan menghapus item yang tidak ada dengan identifier: {identifier}")
        return jsonify({'error': 'Item tidak ditemukan'}), 404

    # Dapatkan path direktori dari variabel global
    folder_mapping = {'uploads': UPLOAD_FOLDER, 'enhanced': ENHANCED_FOLDER}
    
    try:
        if 'original_video_url' in item_to_delete:
            parts = item_to_delete['original_video_url'].strip('/').split('/')
            if len(parts) == 3 and parts[1] in folder_mapping:
                file_path = os.path.join(folder_mapping[parts[1]], parts[2])
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"🗑️ File asli berhasil dihapus: {file_path}")

        if 'enhanced_video_url' in item_to_delete:
            parts = item_to_delete['enhanced_video_url'].strip('/').split('/')
            if len(parts) == 3 and parts[1] in folder_mapping:
                file_path = os.path.join(folder_mapping[parts[1]], parts[2])
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"🗑️ File hasil proses berhasil dihapus: {file_path}")
    except Exception as e:
        logger.error(f"❌ Terjadi kesalahan saat menghapus file untuk identifier {identifier}: {e}")

    del history[index_to_delete]

    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)
    
    logger.info(f"✅ Entri riwayat dengan identifier {identifier} berhasil dihapus")
    return jsonify({'message': 'Item riwayat dan file terkait berhasil dihapus'})


if __name__ == '__main__':
    logger.info("Mengecek ketersediaan model AI...")
    check_and_download_model(SCALE)
    
    # PERUBAHAN: Gunakan port dari environment variable jika ada (untuk Render), jika tidak default ke 5001
    port = int(os.environ.get('PORT', 5001))
    
    # PERUBAHAN: Jangan gunakan debug mode di lingkungan produksi (Render)
    debug_mode = not IS_ON_RENDER
    
    logger.info(f"🚀 Server AI VideoStream siap di http://0.0.0.0:{port} (Debug: {debug_mode})")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)

