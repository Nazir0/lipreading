# Filename: app.py

import os
import json
import glob
import time
import logging
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory, jsonify 
import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
from werkzeug.utils import secure_filename

# --- Configuration & Parameters ---
NUM_FRAMES = 29
IMG_HEIGHT = 64
IMG_WIDTH = 64
CHANNELS = 1
NUM_CLASSES = 60
INPUT_SHAPE = (NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, CHANNELS)
FACE_MESH_MARGIN = 0.3
MOUTH_LANDMARKS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
MOUTH_LANDMARKS_TO_USE = MOUTH_LANDMARKS_OUTER
experiment_suffix = "_tf4x128"
FRONTEND_CONV_FILTERS=32
FRONTEND_KERNEL_SIZE=(5, 5, 5)
FRONTEND_STRIDES=(1, 2, 2)
FRONTEND_POOL_SIZE=(1, 2, 2)
FRONTEND_POOL_STRIDES=(1, 1, 1)
MOBILENET_ALPHA=0.75
TRANSFORMER_EMBED_DIM=128
NUM_TRANSFORMER_BLOCKS=4
NUM_TRANSFORMER_HEADS=8
TRANSFORMER_FF_DIM=512
TRANSFORMER_DROPOUT=0.25
TRANSFORMER_L2_REG=2e-4
USE_ATTENTION_POOLING=True
MODEL_DIR = 'model_files'
BEST_CHECKPOINT_FILENAME = "ckpt-epoch_55-val_acc_0.700.weights.h5" # Adjust if needed
CLASS_MAP_FILENAME = f"lrw_{NUM_CLASSES}words_class_to_int_map.json"
WEIGHTS_PATH = os.path.join(MODEL_DIR, BEST_CHECKPOINT_FILENAME)
CLASS_MAP_PATH = os.path.join(MODEL_DIR, CLASS_MAP_FILENAME)
VIDEO_BASE_PATH = os.path.join('static', 'videos') # Assumes videos in static/videos/WORD/SET/
UPLOAD_FOLDER = 'uploads' # Folder for temporary live uploads

# --- Flask App Setup ---
app = Flask(__name__)
app.secret_key = 'your_very_secret_key' # Change this!
app.logger.setLevel(logging.INFO)

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global variables
lipreading_model = None
int_to_class_map = None
class_to_int_map = None # Store the original map globally too
curated_videos = {} # Dictionary to store display_name: {path: ..., word: ...}

# --- >>> DEFINE YOUR CURATED SAMPLE VIDEOS HERE <<< ---
# Populate this dictionary. Keys are what appear in the dropdown.
# 'path': Path relative to VIDEO_BASE_PATH (e.g., 'static/videos'). Use correct SET folder ('test', 'train', 'val').
# 'word': The correct ground truth word.
SAMPLE_VIDEOS_CONFIG = {
    "BECAUSE": {"path": "BECAUSE/test/BECAUSE_00003_h264.mp4", "word": "BECAUSE"},
    "BETWEEN": {"path": "BETWEEN/test/BETWEEN_00023_h264.mp4", "word": "BETWEEN"},
    "GOING":   {"path": "GOING/test/GOING_00007_h264.mp4", "word": "GOING"},
    "MAJORITY":{"path": "MAJORITY/test/MAJORITY_00019_h264.mp4", "word": "MAJORITY"},
    "PERHAPS": {"path": "PERHAPS/test/PERHAPS_00014_h264.mp4", "word": "PERHAPS"},
    "TIMES":   {"path": "TIMES/test/TIMES_00027_h264.mp4", "word": "TIMES"},
    "YESTERDAY": {"path": "YESTERDAY/test/YESTERDAY_00047_h264.mp4", "word": "YESTERDAY"},
    "ANOTHER": {"path": "ANOTHER/test/ANOTHER_00002_h264.mp4", "word": "ANOTHER"},
    "ANYTHING": {"path": "ANYTHING/test/ANYTHING_00001_h264.mp4", "word": "ANYTHING"},
    "AROUND": {"path": "AROUND/test/AROUND_00001_h264.mp4", "word": "AROUND"},
    "BECOME": {"path": "BECOME/test/BECOME_00001_h264.mp4", "word": "BECOME"},
    "BEING": {"path": "BEING/test/BEING_00001_h264.mp4", "word": "BEING"},
    "EVERY": {"path": "EVERY/test/EVERY_00001_h264.mp4", "word": "EVERY"},
    "EVERYONE": {"path": "EVERYONE/test/EVERYONE_00001_h264.mp4", "word": "EVERYONE"},
    "GETTING": {"path": "GETTING/test/GETTING_00001_h264.mp4", "word": "GETTING"},
    "GIVING": {"path": "GIVING/test/GIVING_00001_h264.mp4", "word": "GIVING"},
    "GOING": {"path": "GOING/test/GOING_00001_h264.mp4", "word": "GOING"},
    "KNOWN": {"path": "KNOWN/test/KNOWN_00001_h264.mp4", "word": "KNOWN"},
    "LITTLE": {"path": "LITTLE/test/LITTLE_00001_h264.mp4", "word": "LITTLE"},
    "LONGER":{"path": "LONGER/test/LONGER_00001_h264.mp4", "word": "LONGER"},
    "LOOKING":{"path": "LOOKING/test/LOOKING_00001_h264.mp4", "word": "LOOKING"},
    "MAKING":{"path": "MAKING/test/MAKING_00001_h264.mp4", "word": "MAKING"},
    "MASSIVE":{"path": "MASSIVE/test/MASSIVE_00001_h264.mp4", "word": "MASSIVE"},
    "SOMETHING":{"path": "SOMETHING/SOMETHING_00001_h264.mp4", "word": "SOMETHING"},
    "TOGETHER":{"path": "TOGETHER/TOGETHER_00001_h264.mp4", "word": "TOGETHER"},
    "WITHIN":{"path": "WITHIN/WITHIN_00001_h264.mp4", "word": "WITHIN"},
    "WITHOUT":{"path": "WITHOUT/WITHOUT_00001_h264.mp4", "word": "WITHOUT"},
    # Add more videos from your 60-word test set if needed
}
# --- >>> END OF SAMPLE VIDEO DEFINITION <<< ---


# --- Model Definition (Layers and build_optimized_visual_classifier function) ---
class AttentionPooling1D(tf.keras.layers.Layer):
    def __init__(self, **kwargs): super().__init__(**kwargs); self.attention_dense = None
    def build(self, input_shape):
        self.attention_dense = tf.keras.layers.Dense(1, activation='tanh', name="att_pool_dense")
        super().build(input_shape)
    def call(self, inputs):
        w_logits = self.attention_dense(inputs)
        w = tf.nn.softmax(w_logits, axis=1)
        return tf.reduce_sum(inputs * w, axis=1)
    def compute_output_shape(self, input_shape):
        return tf.TensorShape((input_shape[0], input_shape[-1]))
    def get_config(self): return super().get_config()

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=sequence_length, output_dim=embed_dim, name="pos_emb_layer")
        self.seq_len = sequence_length; self.vocab_size = vocab_size; self.embed_dim = embed_dim
    def call(self, x):
        L = tf.shape(x)[1]
        positions = tf.range(start=0, limit=L, delta=1)
        positions = tf.minimum(positions, self.seq_len - 1)
        return x + self.pos_emb(positions)
    def get_config(self):
        config = super().get_config(); config.update({"sequence_length": self.seq_len, "vocab_size": self.vocab_size, "embed_dim": self.embed_dim}); return config

class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim; self.num_heads = num_heads; self.ff_dim = ff_dim; self.dropout_rate = dropout_rate; self.l2_reg = l2_reg
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, name="multi_head_attention")
        self.ffn_dense1 = tf.keras.layers.Dense(ff_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name="ffn_dense1")
        self.ffn_dense2 = tf.keras.layers.Dense(embed_dim, kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name="ffn_dense2")
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
    def call(self, inputs, training=False):
        attn_output = self.mha(query=inputs, value=inputs, key=inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn_dense1(out1)
        ffn_output = self.ffn_dense2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    def get_config(self):
        config = super().get_config(); config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads, "ff_dim": self.ff_dim, "dropout_rate": self.dropout_rate, "l2_reg": self.l2_reg}); return config

# Model Build Function (VERIFIED 'inputs' name)
def build_optimized_visual_classifier(
    input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
    frontend_conv_filters=FRONTEND_CONV_FILTERS, frontend_kernel_size=FRONTEND_KERNEL_SIZE,
    frontend_strides=FRONTEND_STRIDES, frontend_pool_size=FRONTEND_POOL_SIZE,
    frontend_pool_strides=FRONTEND_POOL_STRIDES, mobilenet_alpha=MOBILENET_ALPHA,
    transformer_embed_dim=TRANSFORMER_EMBED_DIM, num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
    num_transformer_heads=NUM_TRANSFORMER_HEADS, transformer_ff_dim=TRANSFORMER_FF_DIM,
    transformer_dropout=TRANSFORMER_DROPOUT, transformer_l2_reg=TRANSFORMER_L2_REG,
    use_attention_pooling=USE_ATTENTION_POOLING
    ):
    # Define the input layer and assign it to the variable 'inputs'
    inputs = tf.keras.layers.Input(shape=input_shape, name="video_input", dtype=tf.float32)
    # --- Frontend ---
    x = tf.keras.layers.Conv3D(filters=frontend_conv_filters, kernel_size=frontend_kernel_size, strides=frontend_strides, padding='same', name='frontend_conv3d')(inputs) # Use 'inputs' here
    x = tf.keras.layers.BatchNormalization(name='frontend_bn')(x)
    x = tf.keras.layers.Activation('relu', name='frontend_relu')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=frontend_pool_size, strides=frontend_pool_strides, padding='same', name='frontend_maxpool3d')(x)
    if frontend_conv_filters != 3:
         x = tf.keras.layers.Conv3D(filters=3, kernel_size=(1, 1, 1), padding='same', activation='relu', name='channel_proj_3d')(x)
    # --- Backbone (MobileNetV3Small) ---
    backbone_input_shape = x.shape[2:]
    mobilenet_core = tf.keras.applications.MobileNetV3Small(include_top=False, weights=None, input_shape=backbone_input_shape, pooling='avg', alpha=mobilenet_alpha)
    x = tf.keras.layers.TimeDistributed(mobilenet_core, name='timedist_mobilenet')(x)
    # --- Temporal Projection ---
    mobilenet_output_filters = x.shape[-1]
    if mobilenet_output_filters != transformer_embed_dim:
         x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(transformer_embed_dim, activation='relu'), name='feature_projection')(x)
    # --- Temporal Transformer Backend ---
    x = PositionalEmbedding(NUM_FRAMES, num_classes, transformer_embed_dim, name='positional_embedding')(x)
    x = tf.keras.layers.Dropout(transformer_dropout, name='pos_embedding_dropout')(x)
    for i in range(num_transformer_blocks):
        encoder_layer = TransformerEncoderBlock(
            embed_dim=transformer_embed_dim, num_heads=num_transformer_heads,
            ff_dim=transformer_ff_dim, dropout_rate=transformer_dropout,
            l2_reg=transformer_l2_reg, name=f'transformer_encoder_{i+1}'
        )
        x = encoder_layer(x)
    # --- Temporal Pooling ---
    if use_attention_pooling: x = AttentionPooling1D(name='attention_pooling')(x)
    else: x = tf.keras.layers.GlobalAveragePooling1D(name='temporal_global_avg_pool')(x)
    # --- Classifier Head ---
    # Assign the final tensor to 'outputs'
    outputs = tf.keras.layers.Dense(
        num_classes, activation='softmax', name='classifier_output', dtype='float32',
        kernel_regularizer=tf.keras.regularizers.l2(transformer_l2_reg)
    )(x)
    # Create the model using the 'inputs' and 'outputs' tensors
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=f"lipreading_transformer{experiment_suffix}")
    return model

# --- Preprocessing Functions ---
mp_face_mesh = mp.solutions.face_mesh

def process_frame(frame, face_mesh_instance, margin=FACE_MESH_MARGIN, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    processed_frame = None
    try:
        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = face_mesh_instance.process(img_rgb)
        img_rgb.flags.writeable = True
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            pts = []
            for idx in MOUTH_LANDMARKS_TO_USE:
                if idx < len(face_landmarks.landmark):
                     lm = face_landmarks.landmark[idx]
                     if lm.HasField('x') and lm.HasField('y') and 0 <= lm.x <= 1 and 0 <= lm.y <= 1:
                          pts.append((int(lm.x * w), int(lm.y * h)))
            if len(pts) > 3:
                pts = np.array(pts)
                x_min, y_min = np.min(pts, axis=0)
                x_max, y_max = np.max(pts, axis=0)
                if x_max > x_min and y_max > y_min:
                    center_x = (x_min + x_max) / 2; center_y = (y_min + y_max) / 2
                    crop_size = max(x_max - x_min, y_max - y_min)
                    crop_size_margin = int(crop_size * (1 + 2 * margin))
                    x_min_sq = int(center_x - crop_size_margin / 2); x_max_sq = int(center_x + crop_size_margin / 2)
                    y_min_sq = int(center_y - crop_size_margin / 2); y_max_sq = int(center_y + crop_size_margin / 2)
                    x_min_sq_clipped = max(0, x_min_sq); y_min_sq_clipped = max(0, y_min_sq)
                    x_max_sq_clipped = min(w, x_max_sq); y_max_sq_clipped = min(h, y_max_sq)
                    if y_min_sq_clipped < y_max_sq_clipped and x_min_sq_clipped < x_max_sq_clipped:
                        cropped = frame[y_min_sq_clipped:y_max_sq_clipped, x_min_sq_clipped:x_max_sq_clipped]
                        if cropped.size > 0:
                            resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LINEAR)
                            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                            processed_frame = np.expand_dims(gray, axis=-1).astype(np.uint8)
    except Exception as e:
        app.logger.error(f"Error processing frame: {e}")
        processed_frame = None
    return processed_frame

def preprocess_video(video_path, num_frames=NUM_FRAMES, target_size=(IMG_WIDTH, IMG_HEIGHT), margin=FACE_MESH_MARGIN):
    video_path_str = str(video_path); cap = None; frames = []
    expected_shape = (num_frames, target_size[1], target_size[0], 1)
    last_good_frame = np.full((target_size[1], target_size[0], 1), 128, dtype=np.uint8)
    try:
        cap = cv2.VideoCapture(video_path_str)
        if not cap.isOpened(): raise IOError(f"Could not open video: {video_path_str}")
        total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames_in_video == 0: raise ValueError("Video has 0 frames")
        processed_frame_count = 0; frame_read_count = 0
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5) as face_mesh_instance:
            while processed_frame_count < num_frames:
                ret, frame = cap.read(); frame_read_count += 1
                if not ret: break
                processed = process_frame(frame, face_mesh_instance, margin=margin, target_size=target_size)
                if processed is not None:
                    frames.append(processed); last_good_frame = processed
                else: frames.append(last_good_frame)
                processed_frame_count += 1
        actual_processed_frames = len(frames)
        if actual_processed_frames == 0: raise ValueError("No frames processed")
        elif actual_processed_frames < num_frames:
            for _ in range(num_frames - actual_processed_frames): frames.append(last_good_frame)
        video_array = np.array(frames[:num_frames], dtype=np.uint8)
        if video_array.shape != expected_shape: raise ValueError(f"Final shape mismatch: {video_array.shape}")
        return video_array
    except Exception as e:
        app.logger.error(f"Error during video preprocessing ({video_path_str}): {e}")
        return None
    finally:
        if cap is not None and cap.isOpened(): cap.release()

# --- Load Model and Class Map (Corrected Function V3 - removed expect_partial) ---
def load_model_and_map():
    global lipreading_model, int_to_class_map, curated_videos, class_to_int_map
    if lipreading_model is None:
        app.logger.info("Building model architecture...")
        lipreading_model = build_optimized_visual_classifier()

        try:
            dummy_input = tf.zeros((1,) + INPUT_SHAPE, dtype=tf.float32)
            _ = lipreading_model(dummy_input, training=False)
            app.logger.info("Model built successfully.")
        except Exception as build_e:
            app.logger.error(f"ERROR building model: {build_e}", exc_info=True)
            raise RuntimeError(f"Model architecture failed to build: {build_e}")

        if os.path.exists(WEIGHTS_PATH):
            app.logger.info(f"Loading weights from {WEIGHTS_PATH}...")
            try:
                # Call load_weights without chaining .expect_partial()
                lipreading_model.load_weights(WEIGHTS_PATH)
                app.logger.info("Model weights loaded.")
            except Exception as e:
                 app.logger.error(f"ERROR loading weights: {e}. Check checkpoint compatibility and file path.")
                 raise RuntimeError(f"Could not load weights from {WEIGHTS_PATH}: {e}")
        else:
            app.logger.error(f"Weights file not found at {WEIGHTS_PATH}.")
            raise FileNotFoundError(f"Model weights file not found: {WEIGHTS_PATH}")

    if int_to_class_map is None:
        app.logger.info(f"Loading class map from {CLASS_MAP_PATH}...")
        if os.path.exists(CLASS_MAP_PATH):
            try:
                with open(CLASS_MAP_PATH, 'r') as f:
                    class_to_int_map_loaded = json.load(f)
                class_to_int_map = class_to_int_map_loaded # Store word->int map
                int_to_class_map = {int(v): k for k, v in class_to_int_map_loaded.items()}
                app.logger.info(f"Class map loaded with {len(int_to_class_map)} entries.")
            except Exception as e:
                app.logger.error(f"ERROR loading class map: {e}")
                raise RuntimeError(f"Could not load class map from {CLASS_MAP_PATH}")
        else:
             app.logger.error(f"Class map file not found: {CLASS_MAP_PATH}")
             raise FileNotFoundError(f"Class map file not found: {CLASS_MAP_PATH}")

    if not curated_videos:
        app.logger.info("Populating curated video list...")
        temp_curated = {}
        for display_name, info in SAMPLE_VIDEOS_CONFIG.items():
             # Use the correct subdirectory ('test', 'train', or 'val')
             # Assuming 'test' for now, adjust if needed based on your file structure
             corrected_path = info['path'].replace('/SET/', '/test/') # Make correction here
             file_system_path = os.path.join(VIDEO_BASE_PATH, corrected_path)
             temp_curated[display_name] = {
                 "file_path": file_system_path,
                 "url_path": corrected_path.replace(os.path.sep, '/'),
                 "word": info["word"]
             }
             if not os.path.exists(file_system_path):
                 app.logger.warning(f"Curated video file not found: {file_system_path}")
        curated_videos = temp_curated
        app.logger.info(f"Curated video list populated with {len(curated_videos)} items.")

# --- Utility Functions ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    global curated_videos, class_to_int_map
    try:
        load_model_and_map()
    except Exception as e:
        return f"Application Initialization Error: {e}", 500
    video_options = list(curated_videos.keys())
    word_index_list = []
    if class_to_int_map:
         word_index_list = sorted(class_to_int_map.items(), key=lambda item: item[1])
    return render_template('index.html',
                           video_options=video_options,
                           curated_videos=curated_videos,
                           word_map=word_index_list)

@app.route('/static/videos/<path:filepath>')
def serve_video(filepath):
    video_dir = os.path.abspath(VIDEO_BASE_PATH)
    return send_from_directory(video_dir, filepath)

@app.route('/predict', methods=['POST'])
def predict():
    global lipreading_model, int_to_class_map, curated_videos
    try: load_model_and_map()
    except Exception as e: flash(f"Error initializing: {e}"); return redirect(url_for('index'))
    if int_to_class_map is None or lipreading_model is None: flash("Model/map not loaded."); return redirect(url_for('index'))

    selected_video_key = request.form.get('video_select')
    if not selected_video_key: flash('No video selected.'); return redirect(url_for('index'))
    if selected_video_key not in curated_videos: flash('Invalid video selection.'); return redirect(url_for('index'))

    video_info = curated_videos[selected_video_key]
    video_path = video_info['file_path']; ground_truth_word = video_info['word']
    filename = os.path.basename(video_path)
    app.logger.info(f"Processing: {selected_video_key} ({video_path}), GT: {ground_truth_word}")
    if not os.path.exists(video_path): flash(f'File not found: {filename}'); app.logger.error(f"File not found: {video_path}"); return redirect(url_for('index'))

    try:
        app.logger.info("Preprocessing..."); start_prep = time.time()
        processed_video = preprocess_video(video_path)
        end_prep = time.time()
        if processed_video is None: flash('Preprocessing failed.'); return redirect(url_for('index'))
        app.logger.info(f"Preprocessed in {end_prep - start_prep:.2f}s. Shape: {processed_video.shape}")
        video_normalized = tf.cast(processed_video, tf.float32) / 255.0
        video_batch = tf.expand_dims(video_normalized, axis=0)
        app.logger.info(f"Input shape: {video_batch.shape}")
        app.logger.info("Predicting..."); start_pred = time.time()
        predictions = lipreading_model.predict(video_batch)
        end_pred = time.time()
        app.logger.info(f"Predicted in {end_pred - start_pred:.2f}s.")
        predicted_index = np.argmax(predictions[0])
        predicted_confidence = float(predictions[0][predicted_index])
        predicted_word = int_to_class_map.get(predicted_index, f"Unknown Index: {predicted_index}")
        app.logger.info(f"Prediction: {predicted_word} (Idx: {predicted_index}, Conf: {predicted_confidence:.4f})")
        return render_template('result.html', filename=filename, predicted_word=predicted_word, confidence=f"{predicted_confidence:.4f}", ground_truth_word=ground_truth_word)
    except Exception as e:
        app.logger.error(f"Prediction error for {filename}: {e}", exc_info=True)
        flash(f'Error processing video: {e}')
        return redirect(url_for('index'))

@app.route('/predict_live', methods=['POST'])
def predict_live():
    global lipreading_model, int_to_class_map
    try: load_model_and_map()
    except Exception as e: return jsonify({"error": f"Error initializing: {e}"}), 500
    if int_to_class_map is None or lipreading_model is None: return jsonify({"error": "Model/map not loaded."}), 500

    if 'video_blob' not in request.files: return jsonify({"error": "No video data received."}), 400
    file = request.files['video_blob']
    if file.filename == '': return jsonify({"error": "Received empty file part."}), 400

    # Use a dynamic temporary filename
    temp_filename = f"live_snippet_{int(time.time())}.webm" # Assuming webm from JS recorder
    temp_video_path = os.path.join(UPLOAD_FOLDER, temp_filename)
    if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)

    try:
        file.save(temp_video_path)
        app.logger.info(f"Live snippet saved: {temp_video_path}")
        app.logger.info("Preprocessing live snippet...")
        start_prep = time.time()
        processed_video = preprocess_video(temp_video_path)
        end_prep = time.time()
        if processed_video is None:
             app.logger.error("Live snippet preprocessing failed.")
             return jsonify({"error": "Preprocessing failed."}), 500
        app.logger.info(f"Preprocessed in {end_prep - start_prep:.2f}s. Shape: {processed_video.shape}")

        video_normalized = tf.cast(processed_video, tf.float32) / 255.0
        video_batch = tf.expand_dims(video_normalized, axis=0)
        app.logger.info(f"Input shape: {video_batch.shape}")

        app.logger.info("Predicting live...")
        start_pred = time.time()
        predictions = lipreading_model.predict(video_batch)
        end_pred = time.time()
        app.logger.info(f"Predicted in {end_pred - start_pred:.2f}s.")

        predicted_index = np.argmax(predictions[0])
        predicted_confidence = float(predictions[0][predicted_index])
        predicted_word = int_to_class_map.get(predicted_index, f"Unknown Index: {predicted_index}")
        app.logger.info(f"Live Prediction: {predicted_word} (Idx: {predicted_index}, Conf: {predicted_confidence:.4f})")

        return jsonify({"predicted_word": predicted_word, "confidence": f"{predicted_confidence:.4f}"})
    except Exception as e:
        app.logger.error(f"Live prediction error: {e}", exc_info=True)
        return jsonify({"error": f"Prediction error: {e}"}), 500
    finally:
        if os.path.exists(temp_video_path):
            try: os.remove(temp_video_path); app.logger.info(f"Removed temp snippet: {temp_video_path}")
            except OSError as e_rem: app.logger.error(f"Error removing temp snippet {temp_video_path}: {e_rem}")

# --- Run the App ---
if __name__ == '__main__':
    try:
        load_model_and_map()
        app.logger.info("Model, class map, video list loaded.")
    except Exception as e:
        app.logger.error(f"FATAL startup error: {e}", exc_info=True)
        print(f"Startup error: {e}. Check paths/dependencies. Exiting.")
        exit()

    app.run(debug=True, host='0.0.0.0', port=5001) # Use port 5001